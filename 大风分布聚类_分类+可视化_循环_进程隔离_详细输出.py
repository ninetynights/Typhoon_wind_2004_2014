import os
import re
import math
import sys
import multiprocessing  
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset
import umap.umap_ as umap
import hdbscan
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
import warnings
# 过滤普通警告
warnings.filterwarnings("ignore")
# 过滤多进程资源追踪器的特定警告
warnings.filterwarnings("ignore", module="multiprocessing.resource_tracker")

# ==========================================
#               全局配置区域
# ==========================================

# --- 基础路径配置 (请确保此处包含脚本A和B需要的所有路径) ---
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"

PATHS = {
    # [输入] NC 数据文件
    "nc_file": os.path.join(BASE_DIR, "数据_v2", "Refined_Combine_Stations_ExMaxWind+SLP+StP_Fixed_2004_2024.nc"), # 注意文件名修正
    # [输入] 市界 Shapefile
    "shp_file": os.path.join(BASE_DIR, "地形文件/shapefile/市界/浙江市界.shp"), 
    # [输入] 台风最佳路径文件夹 (txt)
    "best_track_dir": os.path.join(BASE_DIR, "热带气旋最佳路径数据集"),
    # [输入] 大风影响时间 Excel
    "excel_file": os.path.join(BASE_DIR, "数据_v2", "2004_2024_影响台风_大风.xlsx"),
    # [输入] 权威统计详情表 (用于Step2查强度，来自脚本B)
    "stats_file": os.path.join(BASE_DIR, "输出_影响台风特征/统计_1_所有事件详情.csv"),
    
    # [输出] 结果根目录
    "output_base_dir": os.path.join(BASE_DIR, "输出_大风分布聚类/8-10级循环聚类结果")
}

# --- 聚类基本等级配置 ---
CLUSTER_LEVEL_CONFIG = {
    "thresh_min": 17.2,
    "thresh_max": 28.4,
    "name": "8-10级",
    "visual_threshold": 0.5, # 可视化阈值
    "min_samples": 1,        # HDBSCAN min_samples 固定为1
}

# --- 绘图通用设置 ---
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# --- 强度颜色定义 ---
LEVEL_MAP_CH = {
    "0": "弱于热带低压或未知", "1": "热带低压", "2": "热带风暴",
    "3": "强热带风暴", "4": "台风", "5": "强台风", "6": "超强台风", "9": "变性"
}
LEVEL_COLORS_CH = {
    "热带低压": "#A0A0A0", "热带风暴": "#00CED1", "强热带风暴": "#1E90FF",
    "台风": "#FF8C00", "强台风": "#FF4500", "超强台风": "#8B0000",
    "变性": "green", "弱于热带低压或未知": "black",
}
STRENGTH_RANKING_CH = {
    "超强台风": 6, "强台风": 5, "台风": 4, "强热带风暴": 3,
    "热带风暴": 2, "热带低压": 1, "变性": 0, "弱于热带低压或未知": -1
}
ALL_STRENGTH_KEYS_CH = sorted(STRENGTH_RANKING_CH.keys(), key=lambda k: STRENGTH_RANKING_CH[k], reverse=True)

# --- 常量 ---
EXTENT_PATH_MAP = [105, 140, 15, 45] 
COLOR_HL = "#000000"
LW_HL = 1.5

# ==========================================
#           第一部分：辅助函数与类
# ==========================================

@dataclass
class TrackPoint:
    t: pd.Timestamp
    lon: float
    lat: float
    level: str 

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\\s]+", "_", name).strip("_")

def parse_mapping(attr_str: str):
    if not attr_str: return {}
    pairs = (p for p in attr_str.strip().split(";") if ":" in p)
    return {k.strip().zfill(4): v.strip() for k, v in (q.split(":", 1) for q in pairs)}

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
    if omega < 1e-12: return float(lon1), float(lat1)
    so = math.sin(omega)
    v = (math.sin((1-f)*omega)/so)*a + (math.sin(f*omega)/so)*b
    lat = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    lon = math.degrees(math.atan2(v[1], v[0]))
    return lon, lat

def hourly_interp(points: List[TrackPoint]) -> List[TrackPoint]:
    if not points: return []
    points = sorted(points, key=lambda x: x.t)
    times = [p.t for p in points]
    out: List[TrackPoint] = []
    cur = times[0].floor('h')
    end = times[-1].ceil('h')
    idx = 0
    while cur <= end:
        while idx+1 < len(times) and times[idx+1] < cur: idx += 1
        if cur <= times[0]:
            p = points[0]; out.append(TrackPoint(cur, p.lon, p.lat, "Interp"))
        elif cur >= times[-1]:
            p = points[-1]; out.append(TrackPoint(cur, p.lon, p.lat, "Interp"))
        else:
            i = max(0, np.searchsorted(times, cur) - 1); j = i + 1
            t0, t1 = times[i], times[j]
            p0, p1 = points[i], points[j]
            if t0 == t1: out.append(TrackPoint(cur, p0.lon, p0.lat, "Interp"))
            else:
                f = (cur - t0) / (t1 - t0)
                f = float(np.clip(f, 0.0, 1.0))
                lon, lat = slerp_lonlat(p0.lon, p0.lat, p1.lon, p1.lat, f)
                out.append(TrackPoint(cur, lon, lat, "Interp"))
        cur += pd.Timedelta(hours=1)
    return out

def create_combined_view(avg_fp_path, track_path, output_path):
    if not os.path.exists(avg_fp_path) or not os.path.exists(track_path):
        return
    try:
        img1 = mpimg.imread(avg_fp_path)
        img2 = mpimg.imread(track_path)
        fig, axes = plt.subplots(2, 1, figsize=(12, 18))
        axes[0].imshow(img1); axes[0].axis('off')
        axes[1].imshow(img2); axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"   [错误] 拼图失败: {e}")

# ==========================================
#           第二部分：可视化核心函数 (From 脚本B)
# ==========================================

def draw_station_count_text_map(lons, lats, counts, title, out_png, visual_thresh, extent=None):
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=12)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':', edgecolor='gray', alpha=0.6)
    try:
        if os.path.exists(PATHS["shp_file"]):
            city_shapes = list(shpreader.Reader(PATHS["shp_file"]).geometries())
            ax.add_geometries(city_shapes, ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.5, linestyle='--')
    except: pass
    
    # 默认范围
    if extent is None:
        extent = [118, 123, 27, 31.5]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    valid_counts = counts[counts >= visual_thresh]
    vmin = np.nanmin(valid_counts) if len(valid_counts) > 0 else 0.0
    vmax = np.nanmax(counts) if len(counts) > 0 else 1.0
    if vmin == vmax: vmax = vmin + 1.0
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    
    for x, y, val in zip(lons, lats, counts):
        if val < visual_thresh: continue
        color = cmap(norm(val))
        txt = ax.text(x, y, f"{val:.1f}", fontsize=8, ha='center', va='center', color=color, transform=ccrs.PlateCarree())
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="平均统计小时数 (h)")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def setup_map_axis(ax):
    ax.set_extent(EXTENT_PATH_MAP, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax.add_feature(cfeature.BORDERS.with_scale("50m"))
    province_boundaries = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces', scale='10m',
        facecolor='none', edgecolor='grey', linestyle='--', linewidth=0.8
    )
    ax.add_feature(province_boundaries, zorder=3)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)

def draw_legend(ax, level_max_counts: Dict[str, int]):
    legend_handles = []
    for lvl in ALL_STRENGTH_KEYS_CH:
        color = LEVEL_COLORS_CH.get(lvl, "black")
        count = level_max_counts.get(lvl, 0)
        if count > 0:
            label_text = f"{lvl} ({count}个)"
            handle = ax.scatter([], [], color=color, s=30, label=label_text)
            legend_handles.append(handle)
    h_hl = plt.Line2D([0],[0], color=COLOR_HL, lw=LW_HL, label='大风影响时段')
    legend_handles.append(h_hl)
    ax.legend(handles=legend_handles, title="最强强度", loc="lower right", fontsize=8) 

def get_typhoon_ids_from_nc(nc_path: str) -> Set[str]:
    nc = Dataset(nc_path)
    id_attr = nc.getncattr('id_to_index')
    id_map = {k.strip().zfill(4): int(v.strip()) for k, v in [p.split(":") for p in id_attr.split(";") if ":" in p]}
    nc.close()
    return set(id_map.keys())

def read_excel_windows(path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        df = df[["中央台编号", "大风开始时间", "大风结束时间"]].copy()
        df["中央台编号"] = df["中央台编号"].astype(str).str.strip().str.zfill(4)
        df["大风开始时间"] = pd.to_datetime(df["大风开始时间"], errors='coerce')
        df["大风结束时间"] = pd.to_datetime(df["大风结束时间"], errors='coerce')
        return df.dropna().sort_values(["中央台编号", "大风开始时间"])
    except Exception as e:
        # print(f"读取Excel失败: {e}") 
        return pd.DataFrame()

def read_all_tracks_as_points(folder_path: str, valid_ids: Set[str]) -> Dict[str, List[TrackPoint]]:
    track_data = {}
    if not os.path.exists(folder_path): return {}
    for year in range(2004, 2025):
        fname = f"CH{year}BST.txt"
        fpath = os.path.join(folder_path, fname)
        if not os.path.exists(fpath): continue
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            cid, cpath = None, []
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith("66666"):
                    if cid and cid in valid_ids and cpath:
                        track_data.setdefault(cid, []).extend(cpath)
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 5:
                        raw_id = parts[4].strip()
                        cid = raw_id.zfill(4) 
                    else: cid = None
                    cpath = []
                elif cid and cid in valid_ids:
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 6:
                        try:
                            t = pd.to_datetime(parts[0], format="%Y%m%d%H")
                            lvl = LEVEL_MAP_CH.get(parts[1], "弱于热带低压或未知")
                            cpath.append(TrackPoint(t, float(parts[3])/10, float(parts[2])/10, lvl))
                        except: continue
            if cid and cid in valid_ids and cpath:
                track_data.setdefault(cid, []).extend(cpath)
    for k in track_data:
        uniq = {p.t: p for p in track_data[k]}
        track_data[k] = [uniq[t] for t in sorted(uniq.keys())]
    return track_data

def draw_cluster_highlight_map(cluster_id, tids, track_data, impact_windows, level_max_counts, output_path):
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax)

    # 画轨迹线
    for tid in tids:
        if tid not in track_data or not track_data[tid] or tid not in impact_windows: continue
        windows = impact_windows[tid]
        try:
            interpolated_path = hourly_interp(track_data[tid])
        except: continue
        
        for (st, en) in windows:
            xs_h, ys_h = [], []
            for p_interp in interpolated_path:
                if st <= p_interp.t <= en:
                    xs_h.append(p_interp.lon); ys_h.append(p_interp.lat)
            if len(xs_h) >= 2:
                ax.plot(xs_h, ys_h, '-', lw=LW_HL, color=COLOR_HL, alpha=0.9, zorder=2, transform=ccrs.PlateCarree())

    # 画强度点
    for tid in tids:
        if tid not in track_data or not track_data[tid] or tid not in impact_windows: continue
        path = track_data[tid]
        windows = impact_windows[tid]
        for point in path:
            is_in_window = False
            for (st, en) in windows:
                if st <= point.t <= en:
                    is_in_window = True; break
            if is_in_window:
                color = LEVEL_COLORS_CH.get(point.level, "black")
                ax.scatter(point.lon, point.lat, color=color, s=12, transform=ccrs.PlateCarree(), zorder=2)

    draw_legend(ax, level_max_counts)
    total_in_cluster = len(tids)
    ax.set_title(f"聚类 {cluster_id} 路径 (N={total_in_cluster})", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

# ==========================================
#           第三部分：Step 1 & Step 2 融合逻辑
# ==========================================

def run_step1_clustering_enhanced(mcs, nn):
    """
    [修改] 融合了脚本B的高级评估逻辑、UMAP绘图逻辑
    """
    lvl = CLUSTER_LEVEL_CONFIG
    
    # 1. 加载数据
    if not os.path.exists(PATHS['nc_file']):
        return None, None, {"n_clusters": 0, "error": "No NC File"}
        
    nc = Dataset(PATHS['nc_file'])
    id_to_index = parse_mapping(getattr(nc, 'id_to_index', None))
    stids = np.array(nc.variables['STID'][:]).astype(str)
    lats, lons = np.array(nc.variables['lat'][:]), np.array(nc.variables['lon'][:])
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :])
    ty_ids = np.array(nc.variables['typhoon_id_index'][:, 0, :])
    ty_ids = np.where(~np.isnan(ty_ids), ty_ids, -1).astype(int)
    n_sta = wind_speeds.shape[1]
    
    items = sorted([(tid.strip().zfill(4), int(str(idx).strip())) for tid, idx in id_to_index.items()], key=lambda kv: kv[0])
    
    feature_vectors, typhoon_metadata = [], []
    for tid_str, ty_idx in items:
        vec = np.zeros(n_sta)
        for i in range(n_sta):
            mask = (ty_ids[:, i] == ty_idx)
            if not np.any(mask): continue
            ws = wind_speeds[mask, i]
            vec[i] = np.sum((ws >= lvl['thresh_min']) & (ws <= lvl['thresh_max']))
        if np.sum(vec) > 0:
            feature_vectors.append(vec)
            typhoon_metadata.append({"TID": tid_str})

    X = np.array(feature_vectors)
    df_meta = pd.DataFrame(typhoon_metadata)
    # print(f"\n   [信息] 设定为当前的风速范围时，共有 {len(X)} 个台风参与了聚类")
    
    if len(X) < mcs:
         return None, None, {"n_clusters": 0, "error": "Not enough samples"}

    # 2. 聚类流程 [n_jobs=1 保证隔离安全]
    X_scaled = StandardScaler().fit_transform(X)
    X_umap = umap.UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric='euclidean', random_state=42, n_jobs=1).fit_transform(X_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=lvl['min_samples'], metric='euclidean', gen_min_span_tree=True)
    labels = clusterer.fit_predict(X_umap)
    df_meta['Cluster'] = labels
    
    # 3. 计算高级指标 (From 脚本B)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    sil = -1.0
    chi = -1.0
    dbi = -1.0
    dbcv = clusterer.relative_validity_ # HDBSCAN 特有
    
    if n_clusters >= 2:
        core_mask = (labels != -1)
        # 使用 UMAP 降维后的数据计算指标 (与脚本B一致)
        sil = silhouette_score(X_umap[core_mask], labels[core_mask])
        chi = calinski_harabasz_score(X_umap[core_mask], labels[core_mask])
        dbi = davies_bouldin_score(X_umap[core_mask], labels[core_mask])
    
    # 4. 生成包含指标的文件夹名
    safe_name = sanitize_filename(lvl['name'])
    dir_name = (
        f"输出_HDBSCAN_{safe_name}_mcs{mcs}_nn{nn}_"
        f"Sil{sil:.2f}_DBCV{dbcv:.2f}_DBI{dbi:.2f}"
    )
    out_dir = Path(PATHS['output_base_dir']) / dir_name
    ensure_dir(out_dir)
    
    csv_path = out_dir / f"Cluster_Assignments.csv"
    df_meta.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 5. 绘图 (Step 1 增强版)
    # 5.1 平均分布图
    for cid in sorted(set(labels)):
        if cid == -1: continue
        avg_fp = np.mean(X[labels == cid], axis=0)
        title = f"C{cid} (N={np.sum(labels==cid)})"
        draw_station_count_text_map(lons, lats, avg_fp, title, str(out_dir / f"C{cid}_Avg.png"), lvl['visual_threshold'])
        
    # 5.2 UMAP 2D 投影图 (新增)
    umap_2d = umap.UMAP(n_neighbors=nn, n_components=2, min_dist=0.3, random_state=42, n_jobs=1).fit_transform(X_scaled)
    plt.figure(figsize=(10, 8))
    u_labels = np.unique(labels)
    valid_l = u_labels[u_labels != -1]
    cmap_u = plt.cm.get_cmap('Spectral', len(valid_l)) if len(valid_l) > 0 else None
    col_map = {l: cmap_u(i) for i, l in enumerate(valid_l)} if cmap_u else {}
    col_map[-1] = (0.7, 0.7, 0.7, 0.3)
    
    for i, l in enumerate(labels):
        plt.scatter(umap_2d[i, 0], umap_2d[i, 1], color=col_map[l], s=30, alpha=0.8)
    plt.title(f"UMAP Projection (mcs={mcs}, nn={nn})\nSil={sil:.3f}, DBCV={dbcv:.3f}")
    plt.savefig(out_dir / "UMAP_2D_Scatter.png", dpi=150)
    plt.close()

    # 返回所有统计指标
    stats = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels) if len(labels) > 0 else 0,
        "silhouette": sil,
        "dbcv": dbcv,
        "calinski_harabasz": chi,
        "davies_bouldin": dbi,
        "out_dir_name": dir_name # 记录文件夹名方便索引
    }
    return out_dir, csv_path, stats

def run_step2_visualization_full(base_output_dir: Path, cluster_csv_path: Path):
    """
    [修改] 完整的Step2逻辑：读路径、读Excel、画图、拼图
    """
    # 检查依赖文件
    if not os.path.exists(PATHS['stats_file']):
        # 如果没有统计文件，就没法画强度图例，但这不应该阻塞流程
        pass
    
    viz_output_dir = base_output_dir
    
    # 1. 准备数据 (在子进程内读取，虽然耗IO但保证隔离)
    valid_ids = get_typhoon_ids_from_nc(PATHS['nc_file'])
    track_data = read_all_tracks_as_points(PATHS['best_track_dir'], valid_ids)
    excel_df = read_excel_windows(PATHS['excel_file'])
    
    impact_windows = defaultdict(list)
    for _, r in excel_df.iterrows():
        impact_windows[str(r["中央台编号"])].append((r["大风开始时间"], r["大风结束时间"]))
    excel_tids_set = set(impact_windows.keys())

    # 读取强度映射
    id_to_strength = {}
    if os.path.exists(PATHS['stats_file']):
        try:
            df_stats = pd.read_csv(PATHS['stats_file'])
            df_stats['中央台编号'] = df_stats['中央台编号'].astype(str).str.strip().str.zfill(4)
            id_to_strength = dict(zip(df_stats['中央台编号'], df_stats['影响时段最强强度']))
        except: pass

    # 读取聚类结果
    df_c = pd.read_csv(cluster_csv_path, dtype={'TID': str})
    cluster_map = {r['TID'].strip().zfill(4): r['Cluster'] for _, r in df_c.iterrows()}
    clusters = sorted(df_c['Cluster'].unique())

    # 2. 循环绘制
    for cid in clusters:
        if cid == -1: continue # 跳过噪声
        
        tids = [t for t, c in cluster_map.items() if c == cid and t in excel_tids_set]
        if not tids: continue
        
        # 统计强度
        level_max_counts_cluster = defaultdict(int)
        for tid in tids:
            strength = id_to_strength.get(tid, "弱于热带低压或未知")
            level_max_counts_cluster[strength] += 1
        
        # 绘制路径图
        track_png = viz_output_dir / f"Path_Map_C{cid}.png"
        draw_cluster_highlight_map(cid, tids, track_data, impact_windows, level_max_counts_cluster, str(track_png))
        
        # 生成拼图 (Step 1 的图 + Step 2 的图)
        step1_img_path = viz_output_dir / f"C{cid}_Avg.png"
        combined_out = viz_output_dir / f"Combined_View_C{cid}.png"
        create_combined_view(step1_img_path, track_png, combined_out)


# ==========================================
#           第四部分：任务包装与进程启动
# ==========================================

def process_wrapper(mcs, nn):
    """
    单个进程的任务入口
    """
    try:
        out_dir, csv_path, stats = run_step1_clustering_enhanced(mcs, nn)
        if out_dir and csv_path and stats['n_clusters'] > 0:
            # 执行 Step 2
            run_step2_visualization_full(out_dir, csv_path)
        return stats
    except Exception as e:
        # 捕获异常防止进程崩溃导致主循环卡死
        print(f"Error in process (mcs={mcs}, nn={nn}): {e}")
        return {
            "n_clusters": 0, "n_noise": 0, "silhouette": -1, "dbcv": -1, 
            "calinski_harabasz": -1, "davies_bouldin": -1, "error": str(e)
        }

if __name__ == "__main__":
    # --- 参数范围配置 ---
    mcs_range = range(3, 5, 1)  # min_cluster_size
    nn_range = range(2, 3, 1)   # n_neighbors
    
    summary_results = []
    total_tasks = len(mcs_range) * len(nn_range)
    count = 0

    print("="*80)
    print(f" 开始批量全指标评估与可视化")
    print(f" - 总组合数: {total_tasks}")
    print(f" - 结果根目录: {PATHS['output_base_dir']}")
    print("="*80)

    # 确保根目录存在
    Path(PATHS['output_base_dir']).mkdir(parents=True, exist_ok=True)

    for mcs in mcs_range:
        for nn in nn_range:
            count += 1
            print(f"[{count}/{total_tasks}] 运行中: mcs={mcs}, nn={nn} ... ", end="")
            
            # 进程隔离调用
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(1) as pool:
                stats = pool.apply(process_wrapper, (mcs, nn))
            
            # 打印简要结果
            if 'error' in stats and stats['error']:
                print(f"[失败] {stats['error']}")
            else:
                print(f" -> 结果：{stats['n_clusters']}簇 (噪声={stats['n_noise']}个台风, DBCV={stats['dbcv']:.3f}, Sil={stats['silhouette']:.3f}, DBI={stats['davies_bouldin']:.3f})")
            
            # 记录汇总数据
            res = {
                "mcs": mcs, 
                "nn": nn, 
                "n_clusters": stats.get('n_clusters', 0),
                "n_noise": stats.get('n_noise', 0),
                "noise_ratio": stats.get('noise_ratio', 0),
                "silhouette": stats.get('silhouette', -1),
                "dbcv": stats.get('dbcv', -1),
                "calinski_harabasz": stats.get('calinski_harabasz', -1),
                "davies_bouldin": stats.get('davies_bouldin', -1),
                "folder_name": stats.get('out_dir_name', 'Failed')
            }
            summary_results.append(res)

    # 保存超级汇总表
    df_sum = pd.DataFrame(summary_results)
    # 按推荐指标 DBCV 降序排列，方便用户直接看前几行
    # df_sum = df_sum.sort_values(by="dbcv", ascending=False)
    
    summary_path = Path(PATHS['output_base_dir']) / "聚类参数评测汇总表_全指标.csv"
    df_sum.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print(f" 运行完毕！请查看汇总表: {summary_path}")
    print(" 每一行对应的文件夹中均包含完整的路径图、分布图和拼图。")
    print("="*80)