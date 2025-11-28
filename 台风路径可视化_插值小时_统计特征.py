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
import seaborn as sns # [修改 v7] 直接导入，不再检查

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
EXTENT = [100, 145, 15, 45]
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
    # [修复] 增加 .zfill(4) 确保ID如 '414' 变成 '0414'，与txt文件中的格式匹配
    id_map = {k.strip().zfill(4): int(v.strip()) for k, v in [p.split(":") for p in id_attr.split(";") if ":" in p]}
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
    
    for year in range(2004, 2025):
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


# --- 核心绘图函数 (拆分) ---

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
    """[修改 v5] 辅助函数：绘制图例 (按强度等级排序, 使用中文)"""
    legend_handles = []
    # 按照 ALL_STRENGTH_KEYS_CH 的顺序绘制图例
    for lvl in ALL_STRENGTH_KEYS_CH:
        color = LEVEL_COLORS_CH.get(lvl, "black")
        count = level_max_counts.get(lvl, 0)
        # 仅当该等级有统计数据时才显示 (或保留主要热带等级)
        if count > 0 or lvl in ["热带低压", "热带风暴", "强热带风暴", "台风", "强台风", "超强台风"]:
            label_text = f"{lvl} ({count}个)"
            handle = ax.scatter([], [], color=color, s=30, label=label_text)
            legend_handles.append(handle)

    h_hl = plt.Line2D([0],[0], color=COLOR_HL, lw=LW_HL, label='大风影响时段')
    legend_handles.append(h_hl)
    
    ax.legend(handles=legend_handles, title="影响时段内最强强度", loc="lower right", fontsize=8) 
    
def draw_figure_1_combined(
    track_data: Dict[str, List[TrackPoint]], 
    impact_windows: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]], 
    level_max_counts: Dict[str, int],
    excel_tids_set: Set[str],
    output_path: str
):
    """[图1] 绘制整合图：完整路径 + 高亮时段"""
    print("--- 开始绘制 [图1：综合图] ---")
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax)

    print(f"正在绘制 {len(track_data)} 个台风的基础路径...")
    for tid, path in track_data.items():
        if len(path) < 2:
            continue
        lons = [p.lon for p in path]
        lats = [p.lat for p in path]
        ax.plot(lons, lats, color='gray', linewidth=1, alpha=0.6, zorder=1, transform=ccrs.PlateCarree())
        for p in path:
            color = LEVEL_COLORS_CH.get(p.level, "black") 
            ax.scatter(p.lon, p.lat, color=color, s=12, transform=ccrs.PlateCarree(), zorder=2)

    print(f"正在插值并绘制 {len(excel_tids_set)} 个台风的高亮影响时段...")
    for tid, windows in impact_windows.items():
        if tid not in track_data or not track_data[tid]:
            continue
        try:
            interpolated_path = hourly_interp(track_data[tid])
            if not interpolated_path:
                continue
        except Exception as e:
            print(f"警告: 台风 {tid} 插值失败: {e}")
            continue
        for (st, en) in windows:
            xs_h, ys_h = [], []
            for p_interp in interpolated_path:
                if st <= p_interp.t <= en:
                    xs_h.append(p_interp.lon)
                    ys_h.append(p_interp.lat)
            if len(xs_h) >= 2:
                ax.plot(xs_h, ys_h, '-', lw=LW_HL, color=COLOR_HL, alpha=0.9, zorder=2, transform=ccrs.PlateCarree())

    draw_legend(ax, level_max_counts) 
    total_nc = len(track_data)
    ax.set_title(f"2004–2024年影响台风路径 (总数: {total_nc}个)", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"图像 [图1] 已保存至: {output_path}")
    plt.close(fig)

def draw_figure_2_highlight_only(
    track_data: Dict[str, List[TrackPoint]], 
    impact_windows: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]], 
    level_max_counts: Dict[str, int],
    excel_tids_set: Set[str],
    output_path: str
):
    """[图2] 绘制仅高亮图：仅显示影响时段的路径、强度点"""
    print("--- 开始绘制 [图2：仅高亮图] ---")
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax)

    print(f"正在插值并绘制 {len(excel_tids_set)} 个台风的高亮影响时段...")
    for tid, windows in impact_windows.items():
        if tid not in track_data or not track_data[tid]:
            continue
        try:
            interpolated_path = hourly_interp(track_data[tid])
            if not interpolated_path:
                continue
        except Exception as e:
            print(f"警告: 台风 {tid} 插值失败: {e}")
            continue
        for (st, en) in windows:
            xs_h, ys_h = [], []
            for p_interp in interpolated_path:
                if st <= p_interp.t <= en:
                    xs_h.append(p_interp.lon)
                    ys_h.append(p_interp.lat)
            if len(xs_h) >= 2:
                ax.plot(xs_h, ys_h, '-', lw=LW_HL, color=COLOR_HL, alpha=0.9, zorder=2, transform=ccrs.PlateCarree())

    print(f"正在绘制 {len(excel_tids_set)} 个台风在影响时段内的强度点...")
    for tid, windows in impact_windows.items():
        if tid not in track_data or not track_data[tid]:
            continue
        path = track_data[tid]
        for point in path:
            is_in_window = False
            for (st, en) in windows:
                if st <= point.t <= en:
                    is_in_window = True
                    break
            if is_in_window:
                color = LEVEL_COLORS_CH.get(point.level, "black")
                ax.scatter(point.lon, point.lat, color=color, s=12, transform=ccrs.PlateCarree(), zorder=2) # [Bug 修复 v5] p.lat -> point.lat

    draw_legend(ax, level_max_counts) 
    total_excel = len(excel_tids_set)
    ax.set_title(f"2004–2024年台风大风影响时段路径 (共 {total_excel} 个)", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"图像 [图2] 已保存至: {output_path}")
    plt.close(fig)


# --- [新增] 统计分析函数 ---

def analyze_event_features(excel_df_in: pd.DataFrame, track_data: Dict[str, List[TrackPoint]]) -> pd.DataFrame:
    """
    [修改 v5] 分析 Excel 中的每一个“事件”（每一行），计算其时空特征。
    """
    print("--- 开始分析事件时空特征 ---")
    
    ID_COL = "中央台编号"
    ST_COL = "大风开始时间"
    EN_COL = "大风结束时间"
    
    df = excel_df_in.copy()
    if df.empty:
        print("警告: Excel 数据为空，跳过时空特征分析。")
        return pd.DataFrame()

    df['影响时长(h)'] = (df[EN_COL] - df[ST_COL]) / pd.Timedelta(hours=1)
    df['开始年份'] = df[ST_COL].dt.year
    df['开始月份'] = df[ST_COL].dt.month
    df['开始日期'] = df[ST_COL].dt.date

    event_max_levels = []
    for _, row in df.iterrows():
        tid = str(row[ID_COL])
        st = row[ST_COL]
        en = row[EN_COL]
        
        max_strength_rank = -2
        max_level_name = "弱于热带低压或未知" 
        
        if tid in track_data and track_data[tid]:
            path = track_data[tid]
            points_in_window = [p for p in path if st <= p.t <= en]
            
            if points_in_window:
                for p in points_in_window:
                    current_rank = STRENGTH_RANKING_CH.get(p.level, -1) 
                    if current_rank > max_strength_rank:
                        max_strength_rank = current_rank
                        max_level_name = p.level
        event_max_levels.append(max_level_name)
        
    df['影响时段最强强度'] = event_max_levels
    
    print("事件特征分析完成。")
    return df

def generate_statistics_output(analysis_df: pd.DataFrame, output_dir: str):
    """
    [修改 v7] 根据分析好的特征表，生成 CSV 和 PNG 统计报告。
    1. [新增] 强度-总时长柱状图 (Stat #6)
    2. [重编号] 热图编号改为 #7 和 #8
    3. [移除] 移除 seaborn 检查
    """
    if analysis_df.empty:
        print("警告: 分析数据为空，无法生成统计报告。")
        return

    print("\n--- 正在生成统计报告 ---")
    
    total_events = len(analysis_df)

    # --- 绘图颜色配置 (可在此处手动修改) ---
    COLOR_YEAR = "#0000006D"     
    COLOR_MONTH = "#0000006D"       
    COLOR_STRENGTH = "#0000006D"    
    COLOR_DURATION = "#0000006D"   
    COLOR_TOTAL_DURATION = "#0000006D"
    # ------------------------------------------

    # --- 柱顶标签辅助函数 ---
    def add_bar_labels(ax, is_float=False):
        """在 ax 上的每个 bar 顶部添加标签"""
        for bar in ax.patches:
            height = bar.get_height()
            if pd.isna(height) or height == 0:
                continue
            label = f'{height:.1f}' if is_float else f'{height:.0f}'
            
            ax.annotate(label,
                        (bar.get_x() + bar.get_width() / 2, height),
                        ha='center', va='bottom',
                        xytext=(0, 3),
                        textcoords='offset points',
                        fontsize=8.5,
                        color='black')
    # ---------------------------------

    # 1. 保存所有事件的详细表
    output_csv_all = os.path.join(output_dir, "统计_1_所有事件详情.csv")
    try:
        analysis_df.to_csv(output_csv_all, index=False, encoding='utf-8-sig')
        print(f"已保存详细事件表: {output_csv_all}")
    except Exception as e:
        print(f"保存 {output_csv_all} 失败: {e}")

    # 2. 按年份统计
    year_counts = analysis_df['开始年份'].value_counts().sort_index()
    year_counts.name = "事件次数"
    output_csv_year = os.path.join(output_dir, "统计_2_按年统计.csv")
    year_counts.to_csv(output_csv_year)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    year_counts.plot(kind='bar', ax=ax, rot=45, color=COLOR_YEAR)
    ax.set_title(f"2004-2024年浙江省大风影响台风个数年分布 (总个数: {total_events})") 
    ax.set_xlabel("年份")
    ax.set_ylabel("影响台风个数")
    add_bar_labels(ax) 
    plt.tight_layout()
    output_png_year = os.path.join(output_dir, "统计_2_按年统计.png")
    plt.savefig(output_png_year, dpi=150)
    plt.close(fig)
    print(f"已保存按年统计: {output_csv_year} | {output_png_year}")

    # 3. 按月份统计
    month_counts = analysis_df['开始月份'].value_counts().sort_index()
    month_counts.name = "事件次数"
    month_counts.index = month_counts.index.map(lambda x: f"{x}月")
    output_csv_month = os.path.join(output_dir, "统计_3_按月统计.csv")
    month_counts.to_csv(output_csv_month)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    month_counts.reindex([f"{i}月" for i in range(1, 13)]).plot(kind='bar', ax=ax, title="2004-2024年浙江省大风影响台风月分布", rot=0, color=COLOR_MONTH)
    ax.set_title(f"2004-2024年浙江省大风影响台风月分布 (总个数: {total_events})") 
    ax.set_xlabel("月份")
    ax.set_ylabel("台风个数")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    add_bar_labels(ax)
    plt.tight_layout()
    output_png_month = os.path.join(output_dir, "统计_3_按月统计.png")
    plt.savefig(output_png_month, dpi=150)
    plt.close(fig)
    print(f"已保存按月统计: {output_csv_month} | {output_png_month}")

    # 4. 按强度统计 (事件期间的最强强度)
    strength_counts = analysis_df['影响时段最强强度'].value_counts().reindex(ALL_STRENGTH_KEYS_CH).fillna(0).astype(int) 
    strength_counts.name = "事件次数"
    output_csv_strength = os.path.join(output_dir, "统计_4_按强度统计.csv")
    strength_counts.to_csv(output_csv_strength)
    
    fig, ax = plt.subplots(figsize=(10, 7)) 
    strength_counts.plot(kind='bar', ax=ax, rot=45, color=COLOR_STRENGTH)
    ax.set_title(f"2004-2024年浙江省大风影响台风强度 (总个数: {total_events})") 
    ax.set_xlabel("影响时段内的最强强度")
    ax.set_ylabel("台风个数")
    add_bar_labels(ax)
    plt.tight_layout() 
    output_png_strength = os.path.join(output_dir, "统计_4_按强度统计.png")
    plt.savefig(output_png_strength, dpi=150)
    plt.close(fig)
    print(f"已保存按强度统计: {output_csv_strength} | {output_png_strength}")

    # 5. 强度与平均时长结合 (柱状图)
    duration_by_strength = analysis_df.groupby('影响时段最强强度')['影响时长(h)'].mean().reindex(ALL_STRENGTH_KEYS_CH)
    duration_by_strength.name = "平均影响时长(h)"
    output_csv_dur_str = os.path.join(output_dir, "统计_5_强度与平均时长.csv")
    duration_by_strength.to_csv(output_csv_dur_str)
    
    fig, ax = plt.subplots(figsize=(10, 7)) 
    duration_by_strength.dropna().plot(kind='bar', ax=ax, rot=45, color=COLOR_DURATION)
    ax.set_title(f"2004-2024年浙江省大风影响台风\n不同强度台风的平均大风影响时长 (总个数: {total_events})")
    ax.set_xlabel("影响时段内的最强强度")
    ax.set_ylabel("平均影响时长 (小时)")
    add_bar_labels(ax, is_float=True)
    plt.tight_layout()
    output_png_dur_str = os.path.join(output_dir, "统计_5_强度与平均时长.png")
    plt.savefig(output_png_dur_str, dpi=150)
    plt.close(fig)
    print(f"已保存强度与平均时长统计: {output_csv_dur_str} | {output_png_dur_str}")
    
    # --- [新增 v7] 6. 强度与总时长结合 (柱状图) ---
    total_duration_by_strength = analysis_df.groupby('影响时段最强强度')['影响时长(h)'].sum().reindex(ALL_STRENGTH_KEYS_CH).fillna(0)
    total_duration_by_strength.name = "总影响时长(h)"
    output_csv_total_dur_str = os.path.join(output_dir, "统计_6_强度与总时长.csv")
    total_duration_by_strength.to_csv(output_csv_total_dur_str)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    total_duration_by_strength.dropna().plot(kind='bar', ax=ax, rot=45, color=COLOR_TOTAL_DURATION)
    ax.set_title(f"2004-2024年浙江省大风影响台风\n不同强度台风的总大风影响时长 (总个数: {total_events})")
    ax.set_xlabel("影响时段内的最强强度")
    ax.set_ylabel("总影响时长 (小时)")
    add_bar_labels(ax, is_float=True) # 总时长可能小数
    plt.tight_layout()
    output_png_total_dur_str = os.path.join(output_dir, "统计_6_强度与总时长.png")
    plt.savefig(output_png_total_dur_str, dpi=150)
    plt.close(fig)
    print(f"已保存强度与总时长统计: {output_csv_total_dur_str} | {output_png_total_dur_str}")

    # --- [修改 v7] 7 & 8. 热图 (移除 if SEABORN_AVAILABLE 检查) ---
    
    # 7. 强度与月份结合 (热图)
    try:
        strength_by_month = pd.crosstab(analysis_df['开始月份'], analysis_df['影响时段最强强度'])
        ordered_cols = [col for col in ALL_STRENGTH_KEYS_CH if col in strength_by_month.columns]
        strength_by_month = strength_by_month[ordered_cols]
        strength_by_month = strength_by_month.reindex(range(1, 13)).fillna(0).astype(int)
                        
        output_csv_month_str = os.path.join(output_dir, "统计_7_强度与月份热图.csv") # [修改 v7]
        strength_by_month.to_csv(output_csv_month_str)
                        
        fig, ax = plt.subplots(figsize=(12, 8)) 
        sns.heatmap(strength_by_month, annot=True, fmt='d', cmap='YlGnBu', ax=ax, annot_kws={"size": 8})
        ax.set_title(f"2004-2024年浙江省大风影响台风的 月份-强度 分布 (总个数: {total_events})")
        ax.set_xlabel("影响时段内的最强强度")
        ax.set_ylabel("开始月份")
        plt.xticks(rotation=45, ha='right') 
        plt.tight_layout()
        output_png_month_str = os.path.join(output_dir, "统计_7_强度与月份热图.png") # [修改 v7]
        plt.savefig(output_png_month_str, dpi=150)
        plt.close(fig)
        print(f"已保存强度与月份统计: {output_csv_month_str} | {output_png_month_str}")
    except Exception as e:
            print(f"警告: 绘制 [统计_7] 热图失败: {e}")

    # 8. 强度与时长结合 (热图)
    try:
        # 1. 创建时长分箱
        max_dur = analysis_df['影响时长(h)'].max()
        # 确保bins的上限大于或等于最大值
        bins = [0, 24, 48, 72, 96, 120, max(144, max_dur + 1)] 
        labels = ['0-24h', '25-48h', '49-72h', '73-96h', '97-120h', '>120h']
            
        analysis_df_dur = analysis_df[analysis_df['影响时长(h)'] > 0].copy()
        analysis_df_dur['时长分箱'] = pd.cut(analysis_df_dur['影响时长(h)'], bins=bins, labels=labels, right=True)
            
        # 2. 创建交叉表
        strength_by_duration = pd.crosstab(analysis_df_dur['时长分箱'], analysis_df_dur['影响时段最强强度'])
            
        # 3. 排序
        ordered_cols = [col for col in ALL_STRENGTH_KEYS_CH if col in strength_by_duration.columns]
        strength_by_duration = strength_by_duration[ordered_cols]
        strength_by_duration = strength_by_duration.reindex(labels).fillna(0).astype(int)
            
        # 4. 保存CSV
        output_csv_dur_str_hm = os.path.join(output_dir, "统计_8_强度与时长热图.csv") # [修改 v7]
        strength_by_duration.to_csv(output_csv_dur_str_hm)
            
        # 5. 绘图
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(strength_by_duration, annot=True, fmt='d', cmap='Reds', ax=ax, annot_kws={"size": 9})
        ax.set_title(f"2004-2024年浙江省大风影响台风的 强度-时长 分布 (总个数: {total_events})")
        ax.set_xlabel("影响时段内的最强强度")
        ax.set_ylabel("影响时长 (小时)")
        plt.xticks(rotation=45, ha='right') 
        plt.yticks(rotation=0) 
        plt.tight_layout()
        output_png_dur_str_hm = os.path.join(output_dir, "统计_8_强度与时长热图.png") # [修改 v7]
        plt.savefig(output_png_dur_str_hm, dpi=150)
        plt.close(fig)
        print(f"已保存强度与时长热图: {output_csv_dur_str_hm} | {output_png_dur_str_hm}")
            
    except Exception as e:
        print(f"警告: 绘制 [统计_8] 热图失败: {e}")

    print("--- 统计报告生成完毕 ---")


# --- 主流程 ---
if __name__ == "__main__":
    
    # *** 路径配置 ***
    BASE_DIR = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"
    BESTTRACK_DIR = os.path.join(BASE_DIR, "热带气旋最佳路径数据集")
    NC_FILE = os.path.join(BASE_DIR, "数据", "Refined_Combine_Stations_ExMaxWind_Fixed_2004_2024.nc")
    EXCEL_PATH = os.path.join(BASE_DIR, "数据", "2004_2024_影响台风_大风.xlsx")

    OUTPUT_DIR = os.path.join(BASE_DIR, "输出_影响台风特征")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_PNG_1_COMBINED = os.path.join(OUTPUT_DIR, "影响台风路径_1_综合图.png")
    OUTPUT_PNG_2_HIGHLIGHT = os.path.join(OUTPUT_DIR, "影响台风路径_2_仅影响时段.png")

    # 1. [S2] 加载 NC 筛选的台风ID
    valid_typhoon_ids = get_typhoon_ids_from_nc(NC_FILE)
    
    # 2. [S2] 加载这些台风的完整路径数据 (返回中文标签)
    track_data = read_all_tracks_as_points(BESTTRACK_DIR, valid_typhoon_ids)
    
    # 3. [S1] 加载 Excel 影响窗口
    excel_df = read_excel_windows(EXCEL_PATH)
    
    # 3.5 [新增] 对 Excel 中的每一个事件进行时空特征分析 (返回中文标签)
    analysis_df = analyze_event_features(excel_df, track_data)

    # 4. 计算 *图例* 统计数据 (基于 台风ID 的统计)
    impact_windows = defaultdict(list)
    ID_COL = "中央台编号"
    ST_COL = "大风开始时间"
    EN_COL = "大风结束时间"
    for _, r in excel_df.iterrows():
        impact_windows[str(r[ID_COL])].append((r[ST_COL], r[EN_COL]))
    
    excel_tids_set = set(impact_windows.keys())
    
    level_max_counts = defaultdict(int)
    print(f"开始统计 {len(excel_tids_set)} 个Excel台风在影响时段内的最强强度 (用于图例)...")
    
    for tid in excel_tids_set:
        if tid not in track_data or not track_data[tid]:
            level_max_counts["弱于热带低压或未知"] += 1 
            continue
            
        path = track_data[tid]
        windows = impact_windows[tid]
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
        
        level_max_counts[max_level_name] += 1
    print("图例统计完成。")

    # 5. 调用两个绘图函数
    draw_figure_1_combined(
        track_data, impact_windows, level_max_counts, excel_tids_set, OUTPUT_PNG_1_COMBINED
    )
    draw_figure_2_highlight_only(
        track_data, impact_windows, level_max_counts, excel_tids_set, OUTPUT_PNG_2_HIGHLIGHT
    )
    
    # 6. [新增] 生成并保存统计报告 (基于 事件 的统计)
    generate_statistics_output(analysis_df, OUTPUT_DIR)

    print(f"\n--- 所有任务完成 ---")
    print(f"所有输出文件均已保存至: {OUTPUT_DIR}")