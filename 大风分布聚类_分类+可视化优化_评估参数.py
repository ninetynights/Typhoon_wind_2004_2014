import os
import re
import math
import sys
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
# [新增] 引入更多评估指标
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# 过滤普通警告，保持输出清爽
warnings.filterwarnings("ignore")

# ==========================================
#               全局配置区域
# ==========================================

# --- 基础路径配置 ---
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"

PATHS = {
    # [输入] NC 数据文件
    "nc_file": os.path.join(BASE_DIR, "数据_v2", "Refined_Combine_Stations_ExMaxWind+SLP+StP_Fixed_2004_2024.nc"),
    
    # [输入] 市界 Shapefile
    "shp_file": os.path.join(BASE_DIR, "地形文件/shapefile/市界/浙江市界.shp"), 
    
    # [输入] 台风最佳路径文件夹 (txt)
    "best_track_dir": os.path.join(BASE_DIR, "热带气旋最佳路径数据集"),
    
    # [输入] 大风影响时间 Excel
    "excel_file": os.path.join(BASE_DIR, "数据_v2", "2004_2024_影响台风_大风.xlsx"),
    
    # [输入] 权威统计详情表 (用于查强度)
    "stats_file": r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_影响台风特征/统计_1_所有事件详情.csv",

    # [输出] 结果根目录
    "output_base_dir": os.path.join(BASE_DIR, "输出_大风分级统计")
}

# --- 聚类参数配置 (Step 1) ---
CLUSTER_CONFIG = {
    "level": {
        "thresh_min": 28.5,       # 下限
        "thresh_max": 1000,      # 上限
        "name": "11级及以上",      # 名字
    },
    "min_cluster_size": 5,   
    "min_samples": 1,        
    "n_neighbors": 2,       
    "n_components": 5,       
    "min_dist": 0.0,
    "visual_threshold": 0.5
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
    if not os.path.exists(avg_fp_path):
        print(f"   [警告] 无法生成综合图，找不到Step1分布图: {avg_fp_path.name}")
        return
    if not os.path.exists(track_path):
        print(f"   [警告] 无法生成综合图，找不到Step2路径图: {track_path.name}")
        return

    try:
        img1 = mpimg.imread(avg_fp_path)
        img2 = mpimg.imread(track_path)
        fig, axes = plt.subplots(2, 1, figsize=(12, 18))
        axes[0].imshow(img1)
        axes[0].axis('off')
        axes[1].imshow(img2)
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"   [错误] 拼图失败: {e}")

# ==========================================
#           第二部分：Step 1 聚类逻辑
# ==========================================

def draw_station_count_text_map(lons, lats, counts, stids, title, out_png, threshold_val, extent=None, text_size=8, cmap_name="viridis", show_zero=True, vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title + f"\n(平均小时数 < {CLUSTER_CONFIG['visual_threshold']} 已忽略绘制)", fontsize=14)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':', edgecolor='gray', alpha=0.6)
    try:
        if os.path.exists(PATHS["shp_file"]):
            city_shapes = list(shpreader.Reader(PATHS["shp_file"]).geometries())
            ax.add_geometries(city_shapes, ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.5, linestyle='--')
    except Exception: pass
    if extent: ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    valid_counts = counts[counts >= CLUSTER_CONFIG['visual_threshold']]
    if vmin is None: vmin = np.nanmin(valid_counts) if len(valid_counts) > 0 else 0.0
    if vmax is None: vmax = np.nanmax(counts) if len(counts) > 0 else 1.0
    if vmin == vmax: vmax = vmin + 1.0
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    
    for x, y, val in zip(lons, lats, counts):
        if val < CLUSTER_CONFIG['visual_threshold']: continue
        color = cmap(norm(val if np.isfinite(val) else 0.0))
        txt = ax.text(x, y, f"{val:.1f}", fontsize=text_size, ha='center', va='center', color=color, transform=ccrs.PlateCarree())
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="平均统计小时数 (h)")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def run_step1_clustering() -> Tuple[Optional[Path], Optional[Path]]:
    print(f"\n{'='*70}\nSTEP 1: 运行 HDBSCAN + UMAP 聚类分析\n{'='*70}")
    cfg = CLUSTER_CONFIG
    lvl = cfg['level']
    
    # 1. 读取数据
    if not os.path.exists(PATHS['nc_file']):
        print(f"[错误] NC文件不存在: {PATHS['nc_file']}")
        return None, None
    nc = Dataset(PATHS['nc_file'])
    id_to_index = parse_mapping(getattr(nc, 'id_to_index', None))
    index_to_cn = parse_mapping(getattr(nc, 'index_to_cn', None))
    index_to_en = parse_mapping(getattr(nc, 'index_to_en', None))
    stids = np.array(nc.variables['STID'][:]).astype(str)
    lats = np.array(nc.variables['lat'][:], dtype=float)
    lons = np.array(nc.variables['lon'][:], dtype=float)
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    ty_ids = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)
    if np.issubdtype(ty_ids.dtype, np.floating):
        ty_ids = np.where(~np.isnan(ty_ids), ty_ids, -1).astype(int)
    else: ty_ids = ty_ids.astype(int)
    n_time, n_sta = wind_speeds.shape
    
    if id_to_index:
        items = sorted([(tid.strip().zfill(4), int(str(idx).strip())) for tid, idx in id_to_index.items() if str(idx).strip().isdigit()], key=lambda kv: kv[0])
    else:
        uniq = sorted({int(x) for x in np.unique(ty_ids) if int(x) >= 0})
        items = [(str(idx).zfill(4), idx) for idx in uniq]

    # 2. 构建特征
    print(f"构建 {lvl['name']} ({lvl['thresh_min']}~{lvl['thresh_max']}m/s) 特征矩阵...")
    feature_vectors, typhoon_metadata = [], []
    for tid_str, ty_idx in items:
        vec = np.zeros(n_sta, dtype=float)
        for i in range(n_sta):
            mask_ty = (ty_ids[:, i] == ty_idx)
            if not np.any(mask_ty): continue
            ws = wind_speeds[mask_ty, i]
            mask_wind = (ws >= lvl['thresh_min']) & (ws <= lvl['thresh_max'])
            vec[i] = int(np.sum(mask_wind))
        if np.sum(vec) > 0:
            feature_vectors.append(vec)
            typhoon_metadata.append({"TID": tid_str, "Index": ty_idx})
    
    print(f"\n>>> [统计信息] 共有 {len(typhoon_metadata)} 个台风在影响过程中出现了 {lvl['name']} 大风。")

    X = np.array(feature_vectors)
    df_meta = pd.DataFrame(typhoon_metadata)
    
    if X.shape[0] < cfg['min_cluster_size'] * 2:
        print(f"[错误] 有效台风样本数({X.shape[0]})不足(小于2*min_cluster_size)，无法进行有效聚类。")
        return None, None

    # 3. 聚类
    print("执行聚类 (StandardScaler -> UMAP -> HDBSCAN) ...")
    X_scaled = StandardScaler().fit_transform(X)
    X_umap_cluster = umap.UMAP(n_neighbors=cfg['n_neighbors'], n_components=cfg['n_components'], min_dist=cfg['min_dist'], metric='euclidean', random_state=42, n_jobs=1).fit_transform(X_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg['min_cluster_size'], min_samples=cfg['min_samples'], metric='euclidean', gen_min_span_tree=True)
    clusterer.fit(X_umap_cluster)
    labels = clusterer.labels_
    df_meta['Cluster'] = labels
    
    # 4. 结果统计与评估 (重点修改部分)
    cluster_counts = df_meta['Cluster'].value_counts().sort_index()
    n_clusters = len(cluster_counts[cluster_counts.index != -1])
    
    print("\n--- HDBSCAN 聚类结果统计 ---")
    print(cluster_counts)
    print(f"\n总结：找到 {n_clusters} 个主要簇，噪声 {cluster_counts.get(-1, 0)} 个。")

    # --- [计算评估指标] ---
    # 4.1 噪声比率 (越低越好)
    noise_ratio = np.sum(labels == -1) / len(labels)
    
    # 4.2 DBCV (越大越好, HDBSCAN专用)
    dbcv_score = clusterer.relative_validity_
    
    # 4.3 经典指标 (仅对核心簇计算)
    sil_score = -1.0
    chi_score = -1.0
    dbi_score = -1.0
    
    if n_clusters >= 2:
        core_mask = (labels != -1)
        core_X = X_umap_cluster[core_mask] # 使用降维后的数据进行评估
        core_labels = labels[core_mask]
        
        sil_score = silhouette_score(core_X, core_labels)
        chi_score = calinski_harabasz_score(core_X, core_labels)
        dbi_score = davies_bouldin_score(core_X, core_labels)
        
        print(f"\n>>> [综合评估指标]")
        print(f"  1. DBCV (相对有效性): {dbcv_score:.4f} (推荐主要参考)")
        print(f"  2. Silhouette Score: {sil_score:.4f}")
        print(f"  3. Calinski-Harabasz: {chi_score:.1f}")
        print(f"  4. Davies-Bouldin:   {dbi_score:.4f}")
        print(f"  5. 噪声比率:         {noise_ratio:.2%}")
    else:
        print("\n>>> 核心簇不足2个，无法计算轮廓系数等指标。")
    
    # --- 创建结果目录 ---
    safe_lvl_name = sanitize_filename(lvl['name'])
    # 目录名同时包含 Silhouette 和 DBCV
    dir_name = (
        f"输出_台风聚类_HDBSCAN_{safe_lvl_name}_"
        f"mcs{cfg['min_cluster_size']}_"
        f"ms{cfg['min_samples']}_"
        f"nn{cfg['n_neighbors']}_"
        f"sil{sil_score:.2f}_"
        f"dbcv{dbcv_score:.2f}"
        f"dbi{dbi_score:.2f}"
    )
    out_dir = Path(PATHS['output_base_dir']) / dir_name
    ensure_dir(out_dir)
    print(f"结果目录: {out_dir}")
    
    # --- [保存评估结果到 CSV] ---
    eval_data = {
        "Level_Name": lvl['name'],
        "Min_Cluster_Size": cfg['min_cluster_size'],
        "Min_Samples": cfg['min_samples'],
        "N_Neighbors": cfg['n_neighbors'],
        "N_Components": cfg['n_components'],
        "Num_Clusters": n_clusters,
        "Num_Noise": cluster_counts.get(-1, 0),
        "Metric_DBCV": dbcv_score,
        "Metric_Silhouette": sil_score,
        "Metric_Calinski_Harabasz": chi_score,
        "Metric_Davies_Bouldin": dbi_score,
        "Metric_Noise_Ratio": noise_ratio
    }
    pd.DataFrame([eval_data]).to_csv(out_dir / "聚类评估结果.csv", index=False, encoding='utf-8-sig')
    print(f"[OK] 聚类评估结果已保存至: 聚类评估结果.csv")

    # --- 保存聚类明细 ---
    csv_path = out_dir / f"Typhoon_Cluster_Assignments_HDBSCAN_{safe_lvl_name}.csv"
    df_meta.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"正在绘制 {n_clusters} 个聚类的平均分布图...")
    for cid in cluster_counts.index:
        if cid == -1: continue
        c_vecs = X[df_meta[df_meta['Cluster'] == cid].index]
        avg_fp = np.mean(c_vecs, axis=0)
        # 标题显示更多信息
        title = (f"聚类 {cid} (N={len(c_vecs)})\n"
                 f"Sil={sil_score:.2f}, DBCV={dbcv_score:.2f}")
        fname = out_dir / f"Typhoon_Cluster_HDBSCAN_{safe_lvl_name}_C{cid}_AvgFootprint.png"
        draw_station_count_text_map(lons, lats, avg_fp, stids, title, str(fname), lvl['thresh_min'], [118, 123, 27, 31.5])

    # UMAP Plot
    umap_viz = umap.UMAP(n_neighbors=cfg['n_neighbors'], n_components=2, min_dist=0.3, random_state=42).fit_transform(X_scaled)
    plt.figure(figsize=(12, 10))
    u_labels = np.unique(labels)
    valid_l = u_labels[u_labels != -1]
    cmap_u = plt.cm.get_cmap('Spectral', len(valid_l)) if len(valid_l) > 0 else None
    col_map = {l: cmap_u(i) for i, l in enumerate(valid_l)} if cmap_u else {}
    col_map[-1] = (0.7, 0.7, 0.7, 0.5)
    for i, l in enumerate(labels):
        plt.scatter(umap_viz[i, 0], umap_viz[i, 1], color=col_map[l], s=50, alpha=0.8)
        plt.text(umap_viz[i, 0]+0.01, umap_viz[i, 1]+0.01, df_meta.iloc[i]['TID'], fontsize=7)
    
    plt.title(f"UMAP Projection\nSil={sil_score:.3f}, DBCV={dbcv_score:.3f}")
    plt.savefig(out_dir / "Typhoon_Cluster_UMAP_2D_Visualization.png", dpi=180)
    plt.close()
    
    return out_dir, csv_path


# ==========================================
#           第三部分：Step 2 路径绘图逻辑
# ==========================================

# --- 辅助函数：Step 2 专用 ---
def setup_map_axis(ax):
    """辅助函数：设置地图底图"""
    ax.set_extent(EXTENT_PATH_MAP, crs=ccrs.PlateCarree())
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
    """辅助函数：绘制图例"""
    legend_handles = []
    # 按照 ALL_STRENGTH_KEYS_CH 的顺序绘制图例
    for lvl in ALL_STRENGTH_KEYS_CH:
        color = LEVEL_COLORS_CH.get(lvl, "black")
        count = level_max_counts.get(lvl, 0)
        if count > 0:
            label_text = f"{lvl} ({count}个)"
            handle = ax.scatter([], [], color=color, s=30, label=label_text)
            legend_handles.append(handle)

    h_hl = plt.Line2D([0],[0], color=COLOR_HL, lw=LW_HL, label='大风影响时段')
    legend_handles.append(h_hl)
    
    ax.legend(handles=legend_handles, title="影响时段内最强强度", loc="lower right", fontsize=8) 

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
        print(f"读取Excel失败: {e}")
        return pd.DataFrame()

def read_all_tracks_as_points(folder_path: str, valid_ids: Set[str]) -> Dict[str, List[TrackPoint]]:
    track_data = {}
    print(f"读取路径文件: {folder_path}")
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
                    else:
                        cid = None
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

    for tid in tids:
        if tid not in track_data or not track_data[tid] or tid not in impact_windows: continue
        windows = impact_windows[tid]
        try:
            interpolated_path = hourly_interp(track_data[tid])
            if not interpolated_path: continue
        except Exception: continue
        
        for (st, en) in windows:
            xs_h, ys_h = [], []
            for p_interp in interpolated_path:
                if st <= p_interp.t <= en:
                    xs_h.append(p_interp.lon); ys_h.append(p_interp.lat)
            if len(xs_h) >= 2:
                ax.plot(xs_h, ys_h, '-', lw=LW_HL, color=COLOR_HL, alpha=0.9, zorder=2, transform=ccrs.PlateCarree())

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
    ax.set_title(f"2004–2024年台风大风影响时段路径 (聚类: {cluster_id}, 总数: {total_in_cluster}个)", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def run_step2_visualization(base_output_dir: Path, cluster_csv_path: Path):
    STATS_FILE_PATH = PATHS["stats_file"]
    
    print(f"\n{'='*70}\nSTEP 2: 运行路径分类可视化\n输入聚类文件: {cluster_csv_path}\n参考统计文件: {STATS_FILE_PATH}\n{'='*70}")

    if not cluster_csv_path.exists():
        print("[错误] 未找到聚类CSV文件。")
        return
    if not os.path.exists(STATS_FILE_PATH):
        print(f"[错误] 未找到统计详情文件: {STATS_FILE_PATH}")
        return

    viz_output_dir = base_output_dir
    ensure_dir(viz_output_dir)
    
    valid_ids = get_typhoon_ids_from_nc(PATHS['nc_file'])
    track_data = read_all_tracks_as_points(PATHS['best_track_dir'], valid_ids)
    
    excel_df = read_excel_windows(PATHS['excel_file'])
    impact_windows = defaultdict(list)
    for _, r in excel_df.iterrows():
        impact_windows[str(r["中央台编号"])].append((r["大风开始时间"], r["大风结束时间"]))
    excel_tids_set = set(impact_windows.keys())
    
    try:
        df_stats = pd.read_csv(STATS_FILE_PATH)
        df_stats['中央台编号'] = df_stats['中央台编号'].astype(str).str.strip().str.zfill(4)
        id_to_strength = dict(zip(df_stats['中央台编号'], df_stats['影响时段最强强度']))
        print(f"成功加载标准强度表，共 {len(id_to_strength)} 条记录。")
    except Exception as e:
        print(f"[严重错误] 读取统计文件失败: {e}")
        return

    try:
        df_c = pd.read_csv(cluster_csv_path, dtype={'TID': str})
        cluster_map = {
            r['TID'].strip().zfill(4): r['Cluster'] 
            for _, r in df_c.iterrows() 
            if pd.notna(r['Cluster']) and pd.notna(r['TID'])
        }
        clusters = sorted(df_c['Cluster'].unique())
    except Exception as e:
        print(f"聚类CSV读取错误: {e}")
        return

    print(f"开始绘制 {len(clusters)} 个聚类的路径图...")
    for cid in clusters:
        tids = [t for t, c in cluster_map.items() if c == cid and t in excel_tids_set]
        
        all_in_cluster = [t for t, c in cluster_map.items() if c == cid]
        missing_excel = set(all_in_cluster) - set(tids)
        missing_track = [t for t in tids if t not in track_data]

        print(f"\n>> 正在处理 聚类 {cid}:")
        print(f"   - 聚类总数: {len(all_in_cluster)} 个")
        
        if not tids:
            print(f"   - 跳过绘图。")
            continue
            
        level_max_counts_cluster = defaultdict(int)
        for tid in tids:
            strength = id_to_strength.get(tid)
            if strength:
                level_max_counts_cluster[strength] += 1
            else:
                level_max_counts_cluster["弱于热带低压或未知"] += 1
        
        out_png = viz_output_dir / f"台风路径_聚类_{cid}.png"
        draw_cluster_highlight_map(cid, tids, track_data, impact_windows, level_max_counts_cluster, str(out_png))
        print(f"   - [OK] 地图已保存。")

        safe_lvl = sanitize_filename(CLUSTER_CONFIG['level']['name'])
        step1_img_name = f"Typhoon_Cluster_HDBSCAN_{safe_lvl}_C{cid}_AvgFootprint.png"
        step1_img_path = base_output_dir / step1_img_name 
        
        combined_out = viz_output_dir / f"聚类_{cid}.png"
        create_combined_view(step1_img_path, out_png, combined_out)
        if combined_out.exists():
            print(f"   - [OK] 综合视图已生成: {combined_out.name}")
        
    print(f"Step 2 完成。所有路径图保存在: {viz_output_dir}")


if __name__ == "__main__":
    step1_out_dir, step1_csv_path = run_step1_clustering()
    if step1_out_dir and step1_csv_path:
        run_step2_visualization(step1_out_dir, step1_csv_path)
    else:
        print("\n[结束] 由于 Step 1 失败或未生成结果，Step 2 未执行。")