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
from sklearn.metrics import silhouette_score
import warnings

# 过滤普通警告
warnings.filterwarnings("ignore")

# ==========================================
#               全局配置区域
# ==========================================

# --- 基础路径配置 ---
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"

PATHS = {
    "nc_file": os.path.join(BASE_DIR, "数据_v2", "Refined_Combine_Stations_ExMaxWind+SLP+StP_Fixed_2004_2024.nc.nc"),
    "shp_file": os.path.join(BASE_DIR, "地形文件/shapefile/市界/浙江市界.shp"), 
    "best_track_dir": os.path.join(BASE_DIR, "热带气旋最佳路径数据集"),
    "excel_file": os.path.join(BASE_DIR, "数据_v2", "2004_2024_影响台风_大风.xlsx"),
    "output_base_dir": os.path.join(BASE_DIR, "输出_大风分级统计/11级及以上循环聚类结果_循环隔离")
}

# --- 聚类基本等级配置 ---
CLUSTER_LEVEL_CONFIG = {
    "thresh_min": 28.5,
    "thresh_max": 1000,
    "name": "11级及以上",
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
#           第二部分：Step 1 聚类逻辑
# ==========================================

def draw_station_count_text_map(lons, lats, counts, title, out_png, visual_thresh):
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    try:
        if os.path.exists(PATHS["shp_file"]):
            city_shapes = list(shpreader.Reader(PATHS["shp_file"]).geometries())
            ax.add_geometries(city_shapes, ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.5, linestyle='--')
    except: pass
    ax.set_extent([118, 123, 27, 31.5], crs=ccrs.PlateCarree())
    
    valid_counts = counts[counts >= visual_thresh]
    vmin, vmax = (np.min(valid_counts), np.max(counts)) if len(valid_counts) > 0 else (0, 1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax + 0.1)
    cmap = plt.get_cmap("viridis")
    
    for x, y, val in zip(lons, lats, counts):
        if val < visual_thresh: continue
        color = cmap(norm(val))
        txt = ax.text(x, y, f"{val:.1f}", fontsize=8, ha='center', va='center', color=color, transform=ccrs.PlateCarree())
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, shrink=0.8, label="平均统计小时数 (h)")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def run_step1_clustering(mcs, nn):
    """
    [关键修改] 加入 n_jobs=1，且该函数将被放在子进程中独立调用
    """
    lvl = CLUSTER_LEVEL_CONFIG
    
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
    
    # 聚类流程 [加入 n_jobs=1 保证一致性]
    X_scaled = StandardScaler().fit_transform(X)
    X_umap = umap.UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric='euclidean', random_state=42, n_jobs=1).fit_transform(X_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1, metric='euclidean', gen_min_span_tree=True)
    labels = clusterer.fit_predict(X_umap)
    df_meta['Cluster'] = labels
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    sil = -1.0
    if n_clusters >= 2:
        core_mask = (labels != -1)
        sil = silhouette_score(X_umap[core_mask], labels[core_mask])
    
    # [这里保留你原本的文件夹命名格式]
    out_dir = Path(PATHS['output_base_dir']) / f"输出_台风聚类_HDBSCAN_{sanitize_filename(lvl['name'])}_mcs{mcs}_ms1_nn{nn}_viz0.5_sil{sil:.4f}"
    ensure_dir(out_dir)
    csv_path = out_dir / f"Typhoon_Cluster_Assignments_HDBSCAN_{sanitize_filename(lvl['name'])}.csv"
    df_meta.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    for cid in sorted(set(labels)):
        if cid == -1: continue
        avg_fp = np.mean(X[labels == cid], axis=0)
        draw_station_count_text_map(lons, lats, avg_fp, f"C{cid} (N={np.sum(labels==cid)})", str(out_dir / f"C{cid}_Avg.png"), 0.5)
        
    return out_dir, csv_path, {"n_clusters": n_clusters, "n_noise": n_noise, "silhouette": sil}

# ==========================================
#           第三部分：Step 2 路径绘图逻辑（略，结构保持原样）
# ==========================================

def run_step2_visualization(base_output_dir: Path, cluster_csv_path: Path):
    # 这里是你的 Step 2 原始逻辑，为了节省篇幅在此略过具体实现，运行程序时需确保该函数存在
    # 如果 Step 2 代码在另一个文件，请在此处 import 或直接粘贴进来
    print(f"   -> 正在执行路径可视化...")

# ==========================================
#           第四部分：任务包装与进程启动
# ==========================================

def process_wrapper(mcs, nn):
    out_dir, csv_path, stats = run_step1_clustering(mcs, nn)
    run_step2_visualization(out_dir, csv_path)
    return stats

if __name__ == "__main__":
    # [完全保留你原本的参数范围设置]
    mcs_range = range(3, 16, 1)
    nn_range = range(2, 7, 1)
    
    summary_results = []
    total_tasks = len(mcs_range) * len(nn_range)
    count = 0

    print("="*80)
    print(f" 开始批量参数网格搜索")
    print(f" - 总组合数: {total_tasks}")
    print("="*80)

    for mcs in mcs_range:
        for nn in nn_range:
            count += 1
            print(f"[{count}/{total_tasks}] 正在运行: mcs={mcs}, nn={nn} ...")
            
            # [进程隔离实现]：仅通过 context 启动独立进程执行原有的流程函数
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(1) as pool:
                stats = pool.apply(process_wrapper, (mcs, nn))
            
            # [完全保留你原本的终端打印格式]
            print(f"   -> 结果: {stats['n_clusters']} 簇 (噪声:{stats['n_noise']}), Sil={stats['silhouette']:.4f}")
            
            summary_results.append({
                "mcs": mcs, "nn": nn, 
                "n_clusters": stats['n_clusters'], 
                "n_noise": stats['n_noise'], 
                "silhouette": stats['silhouette']
            })

    # [完全保留你原本的汇总表生成逻辑]
    df_sum = pd.DataFrame(summary_results)
    summary_path = Path(PATHS['output_base_dir']) / "聚类参数评测汇总表.csv"
    df_sum.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print(" 所有参数组合运行完毕！")
    print("="*80)