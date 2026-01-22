import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"

# 1. 输入文件路径
# [静态特征]
STATIC_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Station_Static_Features_MultiScale.csv")

# [动态特征]
TRACK_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Typhoon_Tracks_Hourly_Cubic_Features_CubicSpline.csv")

# [标签特征] 聚类结果 (两个文件) - 绝对路径
CLUSTER_CSV_8_10 = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分级统计/输出_台风聚类_HDBSCAN_8-10级_mcs8_ms1_nn2_sil0.74_dbcv0.54dbi0.33/Typhoon_Cluster_Assignments_HDBSCAN_8-10级.csv"
CLUSTER_CSV_11_PLUS = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分级统计/输出_台风聚类_HDBSCAN_11级及以上_mcs9_ms1_nn3_sil0.73_dbcv0.73dbi0.35/Typhoon_Cluster_Assignments_HDBSCAN_11级及以上.csv"

# [观测真值] NC 数据
NC_FILE = os.path.join(BASE_DIR, "数据", "Refined_Combine_Stations_ExMaxWind_Fixed_2004_2024.nc")

# 2. 输出文件
OUTPUT_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Final_Training_Dataset_XGBoost.csv")

# 3. 过滤阈值
MIN_WIND_THRESHOLD = 17.3

# ================= 辅助函数 =================

def haversine_vectorized(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def azimuth_vectorized(lat_st, lon_st, lat_ty, lon_ty):
    lat1, lon1 = np.radians(lat_ty), np.radians(lon_ty)
    lat2, lon2 = np.radians(lat_st), np.radians(lon_st)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def parse_id_mapping(nc_obj):
    """
    更健壮的 ID 解析
    返回两个映射表: 
    1. standard_map: { '0407': index, '2401': index } (标准化4位)
    2. raw_map: { '407': index, ... } (原始字符串)
    """
    attr_str = getattr(nc_obj, 'id_to_index', "")
    pairs = (p for p in attr_str.strip().split(";") if ":" in p)
    
    std_map = {}
    raw_map = {}
    
    for k, v in (q.split(":", 1) for q in pairs):
        idx = int(v.strip())
        raw_key = k.strip()
        
        # 存原始的
        raw_map[raw_key] = idx
        
        # 尝试存标准化的 (4位)
        # 如果 raw_key 是数字 (如 '407'), 补齐为 '0407'
        if raw_key.isdigit():
            std_key = raw_key.zfill(4)
            std_map[std_key] = idx
            
    return std_map, raw_map

# ================= 主流程 =================

def main():
    print(">>> 1. 读取并预处理数据...")
    
    if not os.path.exists(STATIC_CSV):
        print(f"[错误] 找不到静态特征表: {STATIC_CSV}"); return
    df_static = pd.read_csv(STATIC_CSV)
    
    if not os.path.exists(TRACK_CSV):
        print(f"[错误] 找不到路径表: {TRACK_CSV}"); return
    df_track = pd.read_csv(TRACK_CSV, dtype={'TID': str})
    # 强制将 Track ID 标准化为 4 位 (处理 '407' -> '0407')
    df_track['TID'] = df_track['TID'].apply(lambda x: x.zfill(4) if x.isdigit() else x)
    df_track['Time'] = pd.to_datetime(df_track['Time'])
    
    print(f"    - 站点: {len(df_static)}, 路径点: {len(df_track)}")

    # 读取聚类标签
    cluster_map_8_10 = {}
    if os.path.exists(CLUSTER_CSV_8_10):
        df_c1 = pd.read_csv(CLUSTER_CSV_8_10, dtype={'TID': str})
        # 标准化
        df_c1['TID'] = df_c1['TID'].apply(lambda x: x.zfill(4) if x.isdigit() else x)
        cluster_map_8_10 = dict(zip(df_c1['TID'], df_c1['Cluster']))
        
    cluster_map_11 = {}
    if os.path.exists(CLUSTER_CSV_11_PLUS):
        df_c2 = pd.read_csv(CLUSTER_CSV_11_PLUS, dtype={'TID': str})
        df_c2['TID'] = df_c2['TID'].apply(lambda x: x.zfill(4) if x.isdigit() else x)
        cluster_map_11 = dict(zip(df_c2['TID'], df_c2['Cluster']))

    # 读取 NC
    nc = Dataset(NC_FILE)
    nc_stids = np.array(nc.variables['STID'][:]).astype(str)
    
    # [关键修改] 获取 ID 映射 (双重映射表)
    tid_to_idx_std, tid_to_idx_raw = parse_id_mapping(nc)
    
    wind_var = nc.variables['wind_velocity']
    ty_idx_var = nc.variables['typhoon_id_index']
    wind_dims = wind_var.shape
    
    # [关键修复] 读取完整的索引矩阵，不再只读第0个站点
    # 这解决了之前只检查第一个站点导致部分台风被漏判的问题
    print("    - 读取台风索引矩阵 (全量扫描)...")
    if ty_idx_var.ndim == 3:
        # [Time, Level, Station] -> [Time, Station]
        ty_idx_matrix = ty_idx_var[:, 0, :]
    else:
        # [Time, Station]
        ty_idx_matrix = ty_idx_var[:]
        
    # 处理 Masked Array (如果有)
    if np.ma.is_masked(ty_idx_matrix):
        ty_idx_matrix = ty_idx_matrix.filled(-1)
    
    # 确保转为 int 矩阵以便比较
    ty_idx_matrix = ty_idx_matrix.astype(int)
    
    # 静态特征对齐
    valid_mask = np.isin(nc_stids, df_static['STID'])
    valid_stids = nc_stids[valid_mask]
    df_static_indexed = df_static.set_index('STID').reindex(valid_stids)
    
    arr_st_lat = df_static_indexed['Lat'].values
    arr_st_lon = df_static_indexed['Lon'].values
    arr_st_hgt = df_static_indexed['Height'].values
    arr_st_dist = df_static_indexed['Dist_to_Coast'].values
    arr_terrain_5 = df_static_indexed.get('Terrain_Complexity_5km', np.zeros(len(valid_stids))).values
    arr_terrain_10 = df_static_indexed.get('Terrain_Complexity_10km', np.zeros(len(valid_stids))).values
    arr_terrain_15 = df_static_indexed.get('Terrain_Complexity_15km', np.zeros(len(valid_stids))).values

    print(f"    - 静态特征已对齐至 NC 站点顺序 (有效站点数: {len(valid_stids)})")
    
    # ================= 融合循环 =================
    
    all_data_rows = []
    unique_tids = df_track['TID'].unique()
    
    print(f"\n>>> 2. 开始时空融合 (共 {len(unique_tids)} 个台风)...")
    
    processed_count = 0
    skipped_log = {"id_mismatch": [], "no_nc_data": [], "low_wind": []}
    
    for tid in unique_tids:
        # [关键逻辑] ID 匹配尝试
        # 1. 尝试直接匹配标准字典
        nc_idx = tid_to_idx_std.get(tid)
        
        # 2. 如果没找到，尝试去掉前导零匹配 (针对 '0407' -> '407')
        if nc_idx is None and tid.startswith('0'):
            short_tid = tid.lstrip('0')
            nc_idx = tid_to_idx_raw.get(short_tid)
            
        # 3. 还没找到？尝试加上前导零匹配 (针对 '407' -> '0407')
        if nc_idx is None:
            long_tid = tid.zfill(4)
            nc_idx = tid_to_idx_raw.get(long_tid)
            
        if nc_idx is None:
            # print(f"  [跳过] ID不匹配: {tid}")
            skipped_log["id_mismatch"].append(tid)
            continue
            
        # [关键修复] 查找行索引: 只要该行有任意一个站点包含此台风ID，即视为有效
        # 使用 np.any(..., axis=1) 进行行扫描
        row_mask = np.any(ty_idx_matrix == nc_idx, axis=1)
        time_rows_idx = np.where(row_mask)[0]
        
        # 获取路径数据
        track_subset = df_track[df_track['TID'] == tid].sort_values('Time')
        
        n_common = min(len(track_subset), len(time_rows_idx))
        if n_common == 0:
            skipped_log["no_nc_data"].append(tid)
            continue
            
        track_aligned = track_subset.iloc[:n_common]
        nc_rows_aligned = time_rows_idx[:n_common]
        
        # 读取风速
        if len(wind_dims) == 3:
            winds_raw = wind_var[nc_rows_aligned, 0, :] 
        else:
            winds_raw = wind_var[nc_rows_aligned, :]
            
        winds_matrix = np.array(winds_raw)[:, valid_mask]
        
        has_valid_data = False
        
        cluster_8_10 = cluster_map_8_10.get(tid, -1)
        cluster_11 = cluster_map_11.get(tid, -1)
        
        for i in range(n_common):
            row_track = track_aligned.iloc[i]
            obs_winds = winds_matrix[i, :]
            
            filter_mask = (obs_winds >= MIN_WIND_THRESHOLD)
            if not np.any(filter_mask): continue
            
            has_valid_data = True
            
            # 提取数据 (略去部分重复代码，核心逻辑同上)
            batch_df = pd.DataFrame({
                'TID': tid,
                'Time': row_track['Time'],
                'STID': valid_stids[filter_mask],
                'Cluster_8_10': cluster_8_10,
                'Cluster_11_Plus': cluster_11,
                'Obs_Wind_Speed': obs_winds[filter_mask],
                'Ty_Lat': row_track['Lat'],
                'Ty_Lon': row_track['Lon'],
                'Ty_Pressure': row_track['Pressure'],
                'Ty_Center_Wind': row_track['Wind_Speed_Center'],
                'Sta_Lat': arr_st_lat[filter_mask],
                'Sta_Lon': arr_st_lon[filter_mask],
                'Sta_Height': arr_st_hgt[filter_mask],
                'Dist_to_Coast': arr_st_dist[filter_mask],
                'Terrain_5km': arr_terrain_5[filter_mask],
                'Terrain_10km': arr_terrain_10[filter_mask],
                'Terrain_15km': arr_terrain_15[filter_mask],
                'Dist_Station_Ty': haversine_vectorized(arr_st_lon[filter_mask], arr_st_lat[filter_mask], row_track['Lon'], row_track['Lat']),
                'Azimuth_Station_Ty': azimuth_vectorized(arr_st_lat[filter_mask], arr_st_lon[filter_mask], row_track['Lat'], row_track['Lon'])
            })
            all_data_rows.append(batch_df)
            
        if has_valid_data:
            processed_count += 1
        else:
            skipped_log["low_wind"].append(tid)
            
        # [修改] 打印更清晰的进度
        total_scanned = processed_count + len(skipped_log["id_mismatch"]) + len(skipped_log["no_nc_data"]) + len(skipped_log["low_wind"])
        print(f"    已扫描 {total_scanned} / {len(unique_tids)} 个台风... (成功: {processed_count})", end="\r")

    print(f"\n\n>>> 3. 处理报告")
    print(f"    计划处理台风总数: {len(unique_tids)}")
    print(f"    成功入库台风数: {processed_count}")
    print(f"    --- 失败/跳过原因 ---")
    
    if skipped_log['id_mismatch']:
        print(f"    [严重] ID 无法匹配 ({len(skipped_log['id_mismatch'])}个):")
        print(f"      {skipped_log['id_mismatch']}")
        print(f"      (请检查 NC 文件的 id_to_index 属性是否包含这些ID)")
        
    if skipped_log['no_nc_data']:
        print(f"    [严重] 缺 NC 数据/时间未对齐 ({len(skipped_log['no_nc_data'])}个):")
        print(f"      {skipped_log['no_nc_data']}")
        
    if skipped_log['low_wind']:
        print(f"    [警告] 无大风样本 (<{MIN_WIND_THRESHOLD}m/s) ({len(skipped_log['low_wind'])}个):")
        print(f"      {skipped_log['low_wind']}")

    if all_data_rows:
        final_df = pd.concat(all_data_rows, ignore_index=True)
        final_df = final_df.sort_values(['TID', 'Time'])
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n[成功] 文件保存至: {OUTPUT_CSV}")
        print(f"总样本数: {len(final_df)}")
    else:
        print("\n[失败] 未生成任何数据。")

if __name__ == "__main__":
    main()