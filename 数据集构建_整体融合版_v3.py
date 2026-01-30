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
STATIC_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Station_Static_Features_MultiScale_TPI.csv")

# [动态特征]
TRACK_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Typhoon_Tracks_Hourly_Cubic_Features_With_Velocity.csv")

# [标签特征] 聚类结果
CLUSTER_CSV_8_9 = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分布聚类/8-9级循环聚类结果/输出_HDBSCAN_8-9级_mcs7_nn3_Sil0.62_DBCV0.60_DBI0.63/Cluster_Assignments.csv"
CLUSTER_CSV_10_PLUS = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分布聚类/10级及以上循环聚类结果/输出_HDBSCAN_10级及以上_mcs7_nn4_Sil0.76_DBCV0.86_DBI0.29/Cluster_Assignments.csv"

# [观测真值] NC 数据 (包含 StationPress, SeaLevelPress)
NC_FILE = os.path.join(BASE_DIR, "数据_v2", "Refined_Combine_Stations_ExMaxWind+SLP+StP_Fixed_2004_2024.nc")

# 2. 输出文件 (更新为 v3)
OUTPUT_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Final_Training_Dataset_XGBoost_v3_Pressure.csv")

# 3. 过滤阈值
MIN_WIND_THRESHOLD = 17.2

# ================= 辅助函数 =================

def haversine_vectorized(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist = R * c
    # 防止距离为0导致除零错误 (极少数情况台风眼正对测站)
    # 将 0 替换为一个极小值 (如 0.1km)
    dist[dist < 0.1] = 0.1
    return dist

def azimuth_vectorized(lat_st, lon_st, lat_ty, lon_ty):
    lat1, lon1 = np.radians(lat_ty), np.radians(lon_ty)
    lat2, lon2 = np.radians(lat_st), np.radians(lon_st)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def parse_id_mapping(nc_obj):
    attr_str = getattr(nc_obj, 'id_to_index', "")
    pairs = (p for p in attr_str.strip().split(";") if ":" in p)
    std_map = {}
    raw_map = {}
    for k, v in (q.split(":", 1) for q in pairs):
        idx = int(v.strip())
        raw_key = k.strip()
        raw_map[raw_key] = idx
        if raw_key.isdigit():
            std_key = raw_key.zfill(4)
            std_map[std_key] = idx
    return std_map, raw_map

# ================= 主流程 =================

def main():
    print(">>> 1. 读取并预处理数据...")
    
    # 1.1 读取静态特征
    if not os.path.exists(STATIC_CSV):
        print(f"[错误] 找不到静态特征表: {STATIC_CSV}"); return
    df_static = pd.read_csv(STATIC_CSV)
    print(f"    - 已加载静态特征表: {len(df_static)} 行")
    
    # 1.2 读取路径特征
    if not os.path.exists(TRACK_CSV):
        print(f"[错误] 找不到路径表: {TRACK_CSV}"); return
    df_track = pd.read_csv(TRACK_CSV, dtype={'TID': str})
    df_track['TID'] = df_track['TID'].apply(lambda x: x.zfill(4) if x.isdigit() else x)
    df_track['Time'] = pd.to_datetime(df_track['Time'])
    print(f"    - 已加载路径特征表: {len(df_track)} 行")

    # 1.3 读取聚类标签
    cluster_map_8_9 = {}
    if os.path.exists(CLUSTER_CSV_8_9):
        df_c1 = pd.read_csv(CLUSTER_CSV_8_9, dtype={'TID': str})
        df_c1['TID'] = df_c1['TID'].apply(lambda x: x.zfill(4) if x.isdigit() else x)
        cluster_map_8_9 = dict(zip(df_c1['TID'], df_c1['Cluster']))
        
    cluster_map_10 = {}
    if os.path.exists(CLUSTER_CSV_10_PLUS):
        df_c2 = pd.read_csv(CLUSTER_CSV_10_PLUS, dtype={'TID': str})
        df_c2['TID'] = df_c2['TID'].apply(lambda x: x.zfill(4) if x.isdigit() else x)
        cluster_map_10 = dict(zip(df_c2['TID'], df_c2['Cluster']))

    # 1.4 读取 NC 数据 (包含气压变量)
    nc = Dataset(NC_FILE)
    nc_stids = np.array(nc.variables['STID'][:]).astype(str)
    tid_to_idx_std, tid_to_idx_raw = parse_id_mapping(nc)
    
    wind_var = nc.variables['wind_velocity']
    ty_idx_var = nc.variables['typhoon_id_index']
    
    # [关键修改] 读取新增的气压变量
    # 假设变量名是 'StationPress' 和 'SeaLevelPress'，如果不同请在此修改
    # 注意处理可能的命名大小写问题
    keys = nc.variables.keys()
    var_stp_name = next((k for k in keys if k.lower() in ['station_pressure', 'press']), 'StationPress')
    var_slp_name = next((k for k in keys if k.lower() in ['sea_level_pressure', 'mslp']), 'SeaLevelPress')
    
    print(f"    - 正在读取气压变量: 本站气压[{var_stp_name}], 海平面气压[{var_slp_name}]")
    
    var_stp = nc.variables[var_stp_name]
    var_slp = nc.variables[var_slp_name]
    
    wind_dims = wind_var.shape
    
    print("    - 读取台风索引矩阵...")
    if ty_idx_var.ndim == 3:
        ty_idx_matrix = ty_idx_var[:, 0, :]
    else:
        ty_idx_matrix = ty_idx_var[:]
    if np.ma.is_masked(ty_idx_matrix):
        ty_idx_matrix = ty_idx_matrix.filled(-1)
    ty_idx_matrix = ty_idx_matrix.astype(int)
    
    # 1.5 静态特征对齐
    valid_mask = np.isin(nc_stids, df_static['STID'])
    valid_stids = nc_stids[valid_mask]
    df_static_indexed = df_static.set_index('STID').reindex(valid_stids)
    
    # 提取静态向量
    arr_st_lat = df_static_indexed['Lat'].values
    arr_st_lon = df_static_indexed['Lon'].values
    arr_st_hgt = df_static_indexed['Height'].values
    arr_st_dist = df_static_indexed['Dist_to_Coast'].values
    arr_terrain_5 = df_static_indexed.get('Terrain_Complexity_5km', np.zeros(len(valid_stids))).values
    arr_terrain_10 = df_static_indexed.get('Terrain_Complexity_10km', np.zeros(len(valid_stids))).values
    arr_terrain_15 = df_static_indexed.get('Terrain_Complexity_15km', np.zeros(len(valid_stids))).values
    arr_slope = df_static_indexed.get('Slope_Deg', np.zeros(len(valid_stids))).values
    arr_tpi = df_static_indexed.get('TPI', np.zeros(len(valid_stids))).values

    print(f"    - 静态特征对齐完成 (有效站点: {len(valid_stids)})")
    
    # ================= 融合循环 =================
    
    all_data_rows = []
    unique_tids = df_track['TID'].unique()
    
    print(f"\n>>> 2. 开始时空融合 (共 {len(unique_tids)} 个台风)...")
    
    processed_count = 0
    skipped_log = {"id_mismatch": [], "no_nc_data": [], "low_wind": []}
    
    for tid in unique_tids:
        # ID 匹配
        nc_idx = tid_to_idx_std.get(tid)
        if nc_idx is None and tid.startswith('0'):
            short_tid = tid.lstrip('0')
            nc_idx = tid_to_idx_raw.get(short_tid)
        if nc_idx is None:
            long_tid = tid.zfill(4)
            nc_idx = tid_to_idx_raw.get(long_tid)
            
        if nc_idx is None:
            skipped_log["id_mismatch"].append(tid)
            continue
            
        # 行扫描查找
        row_mask = np.any(ty_idx_matrix == nc_idx, axis=1)
        time_rows_idx = np.where(row_mask)[0]
        
        # 路径数据对齐
        track_subset = df_track[df_track['TID'] == tid].sort_values('Time')
        n_common = min(len(track_subset), len(time_rows_idx))
        if n_common == 0:
            skipped_log["no_nc_data"].append(tid)
            continue
            
        track_aligned = track_subset.iloc[:n_common]
        nc_rows_aligned = time_rows_idx[:n_common]
        
        # [关键修改] 读取风速、StP、SLP 数据矩阵
        # 注意：这里我们一次性读取这个台风所有时刻的数据块，然后用 valid_mask 过滤站点
        
        # 1. 风速
        if len(wind_dims) == 3:
            winds_raw = wind_var[nc_rows_aligned, 0, :] 
        else:
            winds_raw = wind_var[nc_rows_aligned, :]
        winds_matrix = np.array(winds_raw)[:, valid_mask]
        
        # 2. 本站气压 (Station Press)
        if var_stp.ndim == 3:
            stp_raw = var_stp[nc_rows_aligned, 0, :]
        else:
            stp_raw = var_stp[nc_rows_aligned, :]
        stp_matrix = np.array(stp_raw)[:, valid_mask]

        # 3. 海平面气压 (Sea Level Press)
        if var_slp.ndim == 3:
            slp_raw = var_slp[nc_rows_aligned, 0, :]
        else:
            slp_raw = var_slp[nc_rows_aligned, :]
        slp_matrix = np.array(slp_raw)[:, valid_mask]
        
        has_valid_data = False
        cluster_8_9 = cluster_map_8_9.get(tid, -1)
        cluster_10 = cluster_map_10.get(tid, -1)
        
        for i in range(n_common):
            row_track = track_aligned.iloc[i]
            obs_winds = winds_matrix[i, :]
            obs_stp = stp_matrix[i, :]
            obs_slp = slp_matrix[i, :]
            
            # 筛选大风样本
            filter_mask = (obs_winds >= MIN_WIND_THRESHOLD)
            if not np.any(filter_mask): continue
            
            has_valid_data = True
            
            # 提取当前时刻的站点坐标 (用于计算距离)
            curr_st_lon = arr_st_lon[filter_mask]
            curr_st_lat = arr_st_lat[filter_mask]
            
            # 计算 站点-台风 距离 (km)
            dist_km = haversine_vectorized(curr_st_lon, curr_st_lat, row_track['Lon'], row_track['Lat'])
            
            # [关键修改] 计算气压梯度特征
            # 公式: (站点气压 - 台风中心气压) / 距离
            # 台风中心气压单位通常是 hPa，NC数据单位也通常是 hPa，直接相减
            ty_press = row_track['Pressure']
            
            # 处理可能的无效气压值 (比如气压传感器故障导致数值异常)
            # 简单的清洗逻辑：如果气压 < 800 或 > 1100，视为无效，填充 NaN
            curr_stp = obs_stp[filter_mask]
            curr_slp = obs_slp[filter_mask]
            
            # 计算梯度 (hPa/km)
            grad_stp = (curr_stp - ty_press) / dist_km
            grad_slp = (curr_slp - ty_press) / dist_km
            
            batch_df = pd.DataFrame({
                # --- 基础信息 ---
                'TID': tid,
                'Time': row_track['Time'],
                'STID': valid_stids[filter_mask],
                'Cluster_8_9': cluster_8_9,
                'Cluster_10_Plus': cluster_10,
                
                # --- 目标标签 ---
                'Obs_Wind_Speed': obs_winds[filter_mask],
                
                # --- 台风动态特征 ---
                'Ty_Lat': row_track['Lat'],
                'Ty_Lon': row_track['Lon'],
                'Ty_Pressure': row_track['Pressure'],
                'Ty_Center_Wind': row_track['Wind_Speed_Center'],
                'Ty_Move_Speed': row_track.get('Move_Speed', np.nan),
                'Ty_Move_Dir': row_track.get('Move_Dir', np.nan),
                
                # --- 站点静态特征 ---
                'Sta_Lat': curr_st_lat,
                'Sta_Lon': curr_st_lon,
                'Sta_Height': arr_st_hgt[filter_mask],
                'Dist_to_Coast': arr_st_dist[filter_mask],
                'Terrain_10km': arr_terrain_10[filter_mask],
                'Sta_Slope': arr_slope[filter_mask],
                'Sta_TPI': arr_tpi[filter_mask],
                
                # --- 相对位置与气压特征 ---
                'Dist_Station_Ty': dist_km,
                'Azimuth_Station_Ty': azimuth_vectorized(curr_st_lat, curr_st_lon, row_track['Lat'], row_track['Lon']),
                
                # [新增] 气压梯度特征
                'Pressure_Gradient_Obs': grad_stp, # 基于本站气压
                'Pressure_Gradient_SLP': grad_slp  # 基于海平面气压
            })
            all_data_rows.append(batch_df)
            
        if has_valid_data:
            processed_count += 1
        else:
            skipped_log["low_wind"].append(tid)
            
        print(f"    已处理 {processed_count} 个台风...", end="\r")

    print(f"\n\n>>> 3. 处理报告")
    print(f"    成功入库台风数: {processed_count}")
    
    if all_data_rows:
        final_df = pd.concat(all_data_rows, ignore_index=True)
        final_df = final_df.sort_values(['TID', 'Time'])
        
        # 简单清洗：去除气压梯度无限大或NaN的情况
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 可以选择是否删除包含 NaN 的行，或者留给后续训练脚本处理
        # final_df.dropna(subset=['Pressure_Gradient_Obs'], inplace=True)
        
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n[成功] 融合数据集已生成: {OUTPUT_CSV}")
        print("前5行预览 (含新特征):")
        print(final_df[['TID', 'Dist_Station_Ty', 'Pressure_Gradient_Obs', 'Pressure_Gradient_SLP']].head())
    else:
        print("\n[失败] 未生成任何数据。")

if __name__ == "__main__":
    main()