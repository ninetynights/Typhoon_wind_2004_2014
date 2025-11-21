import os
import re
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import defaultdict
from netCDF4 import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# --- 强度定义 (中文) ---
LEVEL_MAP_CH = {
    "0": "弱于热带低压或未知",
    "1": "热带低压",
    "2": "热带风暴",
    "3": "强热带风暴",
    "4": "台风",
    "5": "强台风",
    "6": "超强台风",
    "9": "变性"
}

LEVEL_COLORS_CH = {
    "热带低压": "#A0A0A0",
    "热带风暴": "#00CED1",
    "强热带风暴": "#1E90FF",
    "台风": "#FF8C00",
    "强台风": "#FF4500",
    "超强台风": "#8B0000",
    "变性": "green",
    "弱于热带低压或未知": "black",
}

STRENGTH_RANKING_CH = {
    "超强台风": 6,
    "强台风": 5,
    "台风": 4,
    "强热带风暴": 3,
    "热带风暴": 2,
    "热带低压": 1,
    "变性": 0,
    "弱于热带低压或未知": -1
}

# 强度排序键 (从强到弱，包含所有中文)
ALL_STRENGTH_KEYS_CH = sorted(STRENGTH_RANKING_CH.keys(), key=lambda k: STRENGTH_RANKING_CH[k], reverse=True)
# --- 结束强度定义 ---


# --- 插值所需常量 ---
EARTH_R = 6371.0
EXTENT = [105, 140, 15, 45]
COLOR_HL = "#000000"
LW_HL = 1.5


# --- 数据结构 ---
@dataclass
class TrackPoint:
    t: pd.Timestamp
    lon: float
    lat: float
    level: str 

# --- 辅助函数：插值 ---
def slerp_lonlat(lon1, lat1, lon2, lat2, f: float):
    a = np.array([
        math.cos(math.radians(lat1))*math.cos(math.radians(lon1)),
        math.cos(math.radians(lat1))*math.sin(math.radians(lon1)),
        math.sin(math.radians(lat1))
    ])
    b = np.array([
        math.cos(math.radians(lat2))*math.cos(math.radians(lon2)),
        math.cos(math.radians(lat2))*math.sin(math.radians(lon2)),
        math.sin(math.radians(lat2))
    ])
    dot = float(np.clip(np.dot(a,b), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-12:
        return float(lon1), float(lat1)
    so = math.sin(omega)
    v = (math.sin((1-f)*omega)/so)*a + (math.sin(f*omega)/so)*b
    lat = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    lon = math.degrees(math.atan2(v[1], v[0]))
    return lon, lat

def hourly_interp(points: List[TrackPoint]) -> List[TrackPoint]:
    if not points:
        return []
    points = sorted(points, key=lambda x: x.t)
    times = [p.t for p in points]
    out: List[TrackPoint] = []
    cur = times[0].floor('h')
    end = times[-1].ceil('h')
    idx = 0
    
    while cur <= end:
        while idx+1 < len(times) and times[idx+1] < cur:
            idx += 1
        
        if cur <= times[0]:
            p = points[0]
            out.append(TrackPoint(cur, p.lon, p.lat, "Interp"))
        elif cur >= times[-1]:
            p = points[-1]
            out.append(TrackPoint(cur, p.lon, p.lat, "Interp"))
        else:
            i = max(0, np.searchsorted(times, cur) - 1)
            j = i + 1
            t0, t1 = times[i], times[j]
            p0, p1 = points[i], points[j]
            
            if t0 == t1:
                out.append(TrackPoint(cur, p0.lon, p0.lat, "Interp"))
            else:
                f = (cur - t0) / (t1 - t0)
                f = float(np.clip(f, 0.0, 1.0))
                lon, lat = slerp_lonlat(p0.lon, p0.lat, p1.lon, p1.lat, f)
                out.append(TrackPoint(cur, lon, lat, "Interp"))
                
        cur += pd.Timedelta(hours=1)
    return out

# --- 辅助函数：数据读取 ---
def get_typhoon_ids_from_nc(nc_path: str) -> Set[str]:
    nc = Dataset(nc_path)
    id_attr = nc.getncattr('id_to_index')
    id_map = {k.strip(): int(v.strip()) for k, v in [p.split(":") for p in id_attr.split(";") if ":" in p]}
    nc.close()
    print(f"从 {nc_path} 加载了 {len(id_map)} 个台风ID (NC筛选)。")
    return set(id_map.keys())

def read_excel_windows(path: str) -> pd.DataFrame:
    ID_COL = "中央台编号"
    ST_COL = "大风开始时间"
    EN_COL = "大风结束时间"
    try:
        df = pd.read_excel(path)
        df = df[[ID_COL, ST_COL, EN_COL]].copy()
        df[ID_COL] = df[ID_COL].astype(str).str.strip().str.zfill(4)
        df[ST_COL] = pd.to_datetime(df[ST_COL], errors='coerce')
        df[EN_COL] = pd.to_datetime(df[EN_COL], errors='coerce')
        df = df.dropna().sort_values([ID_COL, ST_COL])
        print(f"从 {path} 加载了 {len(df)} 条大风影响事件。")
        return df
    except Exception as e:
        print(f"读取 Excel {path} 失败: {e}")
        return pd.DataFrame(columns=[ID_COL, ST_COL, EN_COL])

def read_all_tracks_as_points(folder_path: str, valid_ids: Set[str]) -> Dict[str, List[TrackPoint]]:
    track_data: Dict[str, List[TrackPoint]] = {}
    print(f"开始从 {folder_path} 目录读取数据，筛选 {len(valid_ids)} 个有效ID...")
    
    for year in range(2010, 2025):
        filename = f"CH{year}BST.txt"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            current_id = None
            current_path: List[TrackPoint] = []

            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("66666"):
                    if current_id and current_id in valid_ids and current_path:
                        if current_id not in track_data:
                            track_data[current_id] = []
                        track_data[current_id].extend(current_path)
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 5:
                        current_id = parts[4]
                        current_path = []
                    else:
                        current_id = None
                        current_path = []
                elif current_id and current_id in valid_ids:
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 6:
                        try:
                            time_str = parts[0]
                            lat = float(parts[2]) / 10.0
                            lon = float(parts[3]) / 10.0
                            level_code = parts[1]
                            level = LEVEL_MAP_CH.get(level_code, "弱于热带低压或未知") 
                            ts = pd.to_datetime(time_str, format="%Y%m%d%H")
                            current_path.append(TrackPoint(ts, lon, lat, level))
                        except Exception:
                            continue

            if current_id and current_id in valid_ids and current_path:
                if current_id not in track_data:
                    track_data[current_id] = []
                track_data[current_id].extend(current_path)

    for tid in track_data:
        uniq = {p.t: p for p in track_data[tid]}
        track_data[tid] = [uniq[t] for t in sorted(uniq.keys())]

    print(f"数据读取完成。共加载 {len(track_data)} 个台风的路径。")
    return track_data


# --- 核心绘图函数 (复用) ---

def setup_map_axis(ax):
    """辅助函数：设置地图底图"""
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax.add_feature(cfeature.BORDERS.with_scale("50m"))
    
    province_boundaries = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces',
        scale='10m',
        facecolor='none',
        edgecolor='grey',
        linestyle='--',
        linewidth=0.8
    )
    ax.add_feature(province_boundaries, zorder=3)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)

def draw_legend(ax, level_max_counts: Dict[str, int]):
    """辅助函数：绘制图例 (按强度等级排序, 使用中文)"""
    legend_handles = []
    # 按照 ALL_STRENGTH_KEYS_CH 的顺序绘制图例
    for lvl in ALL_STRENGTH_KEYS_CH:
        color = LEVEL_COLORS_CH.get(lvl, "black")
        count = level_max_counts.get(lvl, 0)
        # 仅当该等级有统计数据时才显示
        if count > 0:
            label_text = f"{lvl} ({count}个)"
            handle = ax.scatter([], [], color=color, s=30, label=label_text)
            legend_handles.append(handle)

    h_hl = plt.Line2D([0],[0], color=COLOR_HL, lw=LW_HL, label='大风影响时段')
    legend_handles.append(h_hl)
    
    ax.legend(handles=legend_handles, title="影响时段内最强强度", loc="lower right", fontsize=8) 

# --- [新] 聚类绘图函数 ---

def draw_cluster_highlight_map(
    cluster_id: int,
    typhoon_ids_in_cluster: List[str],
    track_data: Dict[str, List[TrackPoint]], 
    impact_windows: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]], 
    level_max_counts: Dict[str, int],
    output_path: str
):
    """[新] 绘制 *单个聚类* 的台风影响时段路径图"""
    
    print(f"--- 开始绘制 [聚类: {cluster_id}] (共 {len(typhoon_ids_in_cluster)} 个台风) ---")
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax)

    # 1. 绘制高亮插值路径
    for tid in typhoon_ids_in_cluster:
        if tid not in track_data or not track_data[tid] or tid not in impact_windows:
            continue
        
        windows = impact_windows[tid]
        
        try:
            interpolated_path = hourly_interp(track_data[tid])
            if not interpolated_path:
                continue
        except Exception as e:
            print(f"警告: 台风 {tid} (聚类 {cluster_id}) 插值失败: {e}")
            continue
        
        for (st, en) in windows:
            xs_h, ys_h = [], []
            for p_interp in interpolated_path:
                if st <= p_interp.t <= en:
                    xs_h.append(p_interp.lon)
                    ys_h.append(p_interp.lat)
            if len(xs_h) >= 2:
                ax.plot(xs_h, ys_h, '-', lw=LW_HL, color=COLOR_HL, alpha=0.9, zorder=2, transform=ccrs.PlateCarree())

    # 2. 绘制影响时段内的强度点
    for tid in typhoon_ids_in_cluster:
        if tid not in track_data or not track_data[tid] or tid not in impact_windows:
            continue
            
        path = track_data[tid]
        windows = impact_windows[tid]
        
        for point in path:
            is_in_window = False
            for (st, en) in windows:
                if st <= point.t <= en:
                    is_in_window = True
                    break
            if is_in_window:
                color = LEVEL_COLORS_CH.get(point.level, "black")
                ax.scatter(point.lon, point.lat, color=color, s=12, transform=ccrs.PlateCarree(), zorder=2)

    # 3. 绘制图例 (使用传入的、针对该聚类计算的 level_max_counts)
    draw_legend(ax, level_max_counts) 
    
    total_in_cluster = len(typhoon_ids_in_cluster)
    ax.set_title(f"2010–2024年台风大风影响时段路径 (聚类: {cluster_id}, 总数: {total_in_cluster}个)", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"图像 [聚类: {cluster_id}] 已保存至: {output_path}")
    plt.close(fig)


# --- 主流程 ---
if __name__ == "__main__":
    
    # *** 路径配置 ***
    # !! 请确保这些路径是正确的 !!
    BASE_DIR = "/Users/momo/Desktop/业务相关/2025 影响台风大风"
    BESTTRACK_DIR = os.path.join(BASE_DIR, "热带气旋最佳路径数据集")
    NC_FILE = os.path.join(BASE_DIR, "数据", "Refined_Combine_Stations_ExMaxWind_Fixed.nc")
    EXCEL_PATH = os.path.join(BASE_DIR, "数据", "2010_2024_影响台风_大风.xlsx")

    # --- [新] 聚类文件路径 ---
    # !! 这是您提供的路径，请确保它准确无误 !!
    CLUSTER_CSV_PATH = "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/输出_台风聚类_Std_UMAP_HDBSCAN_10级及以上_ms2_viz0.1/Typhoon_Cluster_Assignments_HDBSCAN_10级及以上.csv"

    # --- [新] 输出目录 ---
    OUTPUT_DIR_CLUSTER = os.path.join("/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/输出_聚类路径图_10级及以上")
    os.makedirs(OUTPUT_DIR_CLUSTER, exist_ok=True)
    
    # --- [S1] 加载 NC 筛选的台风ID ---
    valid_typhoon_ids = get_typhoon_ids_from_nc(NC_FILE)
    
    # --- [S2] 加载这些台风的完整路径数据 ---
    track_data = read_all_tracks_as_points(BESTTRACK_DIR, valid_typhoon_ids)
    
    # --- [S3] 加载 Excel 影响窗口 ---
    excel_df = read_excel_windows(EXCEL_PATH)
    
    # 将 Excel 窗口转为字典，方便查询
    impact_windows = defaultdict(list)
    ID_COL = "中央台编号"
    ST_COL = "大风开始时间"
    EN_COL = "大风结束时间"
    for _, r in excel_df.iterrows():
        impact_windows[str(r[ID_COL])].append((r[ST_COL], r[EN_COL]))
    
    # 具有大风影响事件的台风ID集合
    excel_tids_set = set(impact_windows.keys())
    
    # --- [S4] 加载聚类CSV文件 ---
    # --- [S4] 加载聚类CSV文件 ---
    print(f"--- 正在加载聚类文件: {CLUSTER_CSV_PATH} ---")
    try:
        # !! 关键假设 !!
        # 假设台风ID列名为 'TID'，聚类ID列名为 'Cluster'
        # 并且 'TID' 是 4位字符串
        TID_COL_NAME = 'TID'
        CLUSTER_COL_NAME = 'Cluster'
        
        df_cluster = pd.read_csv(CLUSTER_CSV_PATH, dtype={TID_COL_NAME: str})
        
        # 创建一个 台风ID -> 聚类ID 的映射字典
        # zfill(4) 确保 '2001' 和 '2001' 能匹配上
        
        # --- [修正] ---
        # 错误：tid.strip().zfill(4): row[CLUSTER_COL_NAME]
        # 正确：row[TID_COL_NAME].strip().zfill(4): row[CLUSTER_COL_NAME]
        cluster_map = {
            row[TID_COL_NAME].strip().zfill(4): row[CLUSTER_COL_NAME] 
            for _, row in df_cluster.iterrows() 
            if pd.notna(row[CLUSTER_COL_NAME]) and pd.notna(row[TID_COL_NAME]) and row[TID_COL_NAME]
        }
        # --- [修正结束] ---
        
        unique_clusters = sorted(df_cluster[CLUSTER_COL_NAME].unique())
        
        print(f"加载聚类文件成功。共 {len(cluster_map)} 个台风被分配了聚类。")
        print(f"发现 {len(unique_clusters)} 个唯一聚类: {unique_clusters}")
        
    except Exception as e:
        print(f"!!! 严重错误: 无法读取或解析聚类文件 {CLUSTER_CSV_PATH} !!!")
        print(f"错误详情: {e}")
        print("请检查文件路径、文件是否存在、以及列名是否为 'TID' 和 'Cluster'。")
        exit() # 退出脚本，因为没有聚类无法继续

    
    # --- [S5] 按聚类循环绘图 ---
    print("\n--- 开始按聚类循环绘制路径图 ---")
    
    for cluster_id in unique_clusters:
        # 1. 筛选出属于当前聚类的台风ID
        typhoon_ids_in_cluster = [
            tid for tid, cid in cluster_map.items() if cid == cluster_id
        ]
        
        # 2. 进一步筛选：只保留那些 *同时* 存在于 Excel 大风事件中的台风
        tids_to_plot = [
            tid for tid in typhoon_ids_in_cluster if tid in excel_tids_set
        ]
        
        if not tids_to_plot:
            print(f"聚类 {cluster_id} 中没有台风有对应的大风事件，跳过绘图。")
            continue
            
        # 3. [关键] 为 *这个聚类* 计算专属的图例统计数据
        #    (复制原脚本中的图例统计逻辑，但只对 tids_to_plot 运行)
        level_max_counts_cluster = defaultdict(int)
        
        for tid in tids_to_plot:
            if tid not in track_data or not track_data[tid]:
                level_max_counts_cluster["弱于热带低压或未知"] += 1 
                continue
                
            path = track_data[tid]
            windows = impact_windows[tid] # 我们已知 tid 在 impact_windows 中
            max_strength_rank = -2
            max_level_name = "弱于热带低压或未知" 
            
            for point in path:
                is_in_window = False
                for (st, en) in windows:
                    if st <= point.t <= en:
                        is_in_window = True
                        break
                if is_in_window:
                    current_rank = STRENGTH_RANKING_CH.get(point.level, -1) 
                    if current_rank > max_strength_rank:
                        max_strength_rank = current_rank
                        max_level_name = point.level
            
            level_max_counts_cluster[max_level_name] += 1
        
        # 4. 定义输出路径
        output_png_path = os.path.join(OUTPUT_DIR_CLUSTER, f"台风路径_聚类_{cluster_id}.png")
        
        # 5. 调用绘图函数
        draw_cluster_highlight_map(
            cluster_id,
            tids_to_plot,
            track_data,
            impact_windows,
            level_max_counts_cluster, # 传入该聚类专属的统计
            output_png_path
        )

    print(f"\n--- 所有聚类绘图任务完成 ---")
    print(f"所有聚类地图均已保存至: {OUTPUT_DIR_CLUSTER}")