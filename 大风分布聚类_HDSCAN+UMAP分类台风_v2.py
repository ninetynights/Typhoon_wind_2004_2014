"""
大风分布聚类_HDSCAN+UMAP分类台风_v2.py

目的与功能（v2 扩展版）：
- 与 v1 相同的总体目标（基于台风对站点的大风小时数做聚类），但在预处理、配置与可视化上做了若干增强：
  * 增加 StandardScaler 标准化（与 v1 一致），并允许自定义 UMAP/HDBSCAN 超参；
  * 引入 min_samples 参数（MIN_SAMPLES_PARAM）以更灵活控制 HDBSCAN；
  * 增加 VISUAL_THRESHOLD：绘图时可过滤掉“很小”的平均小时数（例如 <0.1），减少地图噪声；
  * 改善输出目录与文件名的可读性（sanitize_filename 与包含参数的子目录名）；
  * 运行中打印更丰富的任务配置与运行信息。

主要输入：
- NC_PATH：同 v1，包含台风-站点-小时矩阵。
- SHP_CITY_PATH：市界 Shapefile 路径（绘图参考，用于叠加市界边界）。
- BASE_OUTPUT_DIR：结果输出根目录。
- LEVEL_CONFIG：指定待分析的风速级别阈值（thresh_min/thresh_max/name）。

主要输出（保存到 BASE_OUTPUT_DIR/输出_台风聚类... 子目录）：
- Typhoon_Cluster_Assignments_HDBSCAN_{safe_level_name}.csv：台风聚类分配表。
- Typhoon_Cluster_HDBSCAN_{safe_level_name}_C{簇}_AvgFootprint.png：按簇输出的平均分布地图（已过滤低值）。
- Typhoon_Cluster_UMAP_2D_Visualization.png：UMAP 2D 可视化散点图。
- 控制台打印：聚类统计、轮廓系数（若可计算）及运行参数摘要。

- 可视化阈值 VISUAL_THRESHOLD 控制地图上是否显示较小值，避免图上过多零或微小数值干扰。
- 输出目录名称包含关键参数，便于批量试验与结果管理。
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset
import umap.umap_ as umap
import hdbscan

# --- 导入 StandardScaler ---
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# ======= 1. NetCDF 路径 =======
NC_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/Refined_Combine_Stations_ExMaxWind_Fixed.nc"

# ======= 2. Shapefile 路径 =======
SHP_CITY_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/shapefile/市界/浙江市界.shp"

# ======= 3. 基础输出目录 =======
BASE_OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计")

# ======= 4. 【新】HDBSCAN + UMAP 聚类任务配置 =======

# --- 我们要分析哪个级别？---
LEVEL_CONFIG = {
    "thresh_min": 17.2,
    "thresh_max": 24.4,
    "name": "8-9级",
}

# --- 【关键】HDBSCAN 参数设定 ---
MIN_CLUSTER_SIZE = 4 
MIN_SAMPLES_PARAM = 2 # 最终调优参数

# --- 【关键】UMAP 参数设定 ---
N_COMPONENTS_CLUSTER = 5
N_NEIGHBORS = 8
MIN_DIST = 0.0

# --- 【新】可视化阈值 ---
VISUAL_THRESHOLD = 0.1 # 你的新想法！

# --- 绘图参数 (不变) ---
EXTENT = [118, 123, 27, 31.5]
TEXT_SIZE = 8
CMAP_NAME = "viridis"
SHOW_ZERO = False 

# ----------------------------- 辅助函数 (绘图, 已修改) -----------------------------
def parse_mapping(attr_str: str):
    if not attr_str:
        return {}
    pairs = (p for p in attr_str.strip().split(";") if ":" in p)
    return {k.strip(): v.strip() for k, v in (q.split(":", 1) for q in pairs)}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    # 确保此函数在 `main` 之前定义
    return re.sub(r"[\\/:*?\"<>|\\s]+", "_", name).strip("_")

def draw_station_count_text_map(
    lons, lats, counts, stids, title, out_png,
    threshold_val, 
    extent=None, text_size=8, cmap_name="viridis", show_zero=True,
    vmin=None, vmax=None
):
    """
    (函数已修改，加入了你的 0.5 阈值)
    """
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 【修改】在标题中加入阈值信息
    ax.set_title(title + f"\n(平均小时数 < {VISUAL_THRESHOLD} 已忽略绘制)", fontsize=14)
    
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    try:
        city_shapes = list(shpreader.Reader(SHP_CITY_PATH).geometries())
        ax.add_geometries(
            city_shapes, ccrs.PlateCarree(),
            edgecolor='gray', facecolor='none',
            linewidth=0.5, linestyle='--'
        )
    except Exception as e:
        print(f"\n[WARN] 无法加载市界 Shapefile: {SHP_CITY_PATH}")
        pass
    if extent and len(extent) == 4:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons)-pad_lon, np.nanmax(lons)+pad_lon,
                       np.nanmin(lats)-pad_lat, np.nanmax(lats)+pad_lat],
                      crs=ccrs.PlateCarree())
    
    # 过滤掉低于阈值的值，以便正确计算色标
    valid_counts = counts[counts >= VISUAL_THRESHOLD]
    if vmin is None:
        vmin = np.nanmin(valid_counts) if len(valid_counts) > 0 else 0.0
    if vmax is None:
        vmax = np.nanmax(counts) if len(counts) > 0 else 1.0
        
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
        
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    
    # --- 【【【【【核心可视化修改】】】】】 ---
    for x, y, val in zip(lons, lats, counts):
        
        # 你的新想法：只显示 0.5 以上的值
        if val < VISUAL_THRESHOLD:
            continue
        # --- 【【【【【修改结束】】】】】 ---

        color = cmap(norm(val if np.isfinite(val) else 0.0))
        
        display_val = f"{val:.1f}" 
             
        txt = ax.text(
            x, y, display_val, fontsize=text_size,
            ha='center', va='center', color=color,
            transform=ccrs.PlateCarree()
        )
        txt.set_path_effects([
            pe.withStroke(linewidth=1.5, foreground="white")
        ])
    # --- 【【【【【循环结束】】】】】 ---
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(f"平均统计小时数 (m/s)")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, crs=ccrs.PlateCarree())
    gl.right_labels = False
    gl.top_labels = False
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ----------------------------- 主逻辑 (UMAP + HDBSCAN) -----------------------------
def main():
    
    # 1. 定义配置和输出目录
    config = LEVEL_CONFIG
    level_name = config['name']
    thresh_min = config['thresh_min']
    thresh_max = config['thresh_max']
    
    # --- 【【【【【核心命名修改】】】】】 ---
    # 我们把输出目录改一下名，以作区分
    safe_level_name = sanitize_filename(level_name) # <-- 先获取安全名称
    output_subdir_name = f"输出_台风聚类_Std_UMAP_HDBSCAN_{safe_level_name}_ms{MIN_SAMPLES_PARAM}_viz{VISUAL_THRESHOLD}"
    output_dir = BASE_OUTPUT_DIR / output_subdir_name
    # --- 【【【【【修改结束】】】】】 ---
    
    ensure_dir(output_dir)
    
    print(f"{'='*70}")
    print(f"--- 任务：StandardScaler + UMAP + HDBSCAN 台风聚类 (最终可视化版) ---")
    print(f"--- 级别：“{level_name}” ---")
    print(f"--- HDBSCAN min_cluster_size = {MIN_CLUSTER_SIZE} ---")
    print(f"--- HDBSCAN min_samples = {MIN_SAMPLES_PARAM} (最终调优) ---")
    print(f"--- 可视化阈值 = {VISUAL_THRESHOLD} (新) ---")
    print(f"--- 输出目录: {output_dir.resolve()} ---")
    print(f"{'='*70}")

    # 2. 读数据 (不变)
    print(f"正在读取 NetCDF 文件: {NC_PATH}")
    nc = Dataset(NC_PATH)

    id_to_index = parse_mapping(getattr(nc, 'id_to_index', None))
    index_to_cn = parse_mapping(getattr(nc, 'index_to_cn', None))
    index_to_en = parse_mapping(getattr(nc, 'index_to_en', None))

    stids = np.array(nc.variables['STID'][:]).astype(str)
    lats = np.array(nc.variables['lat'][:], dtype=float)
    lons = np.array(nc.variables['lon'][:], dtype=float)
    
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    ty_ids = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)
    if np.issubdtype(ty_ids.dtype, np.floating):
        ty_ids_int = np.full_like(ty_ids, fill_value=-1, dtype=int)
        valid = ~np.isnan(ty_ids)
        ty_ids_int[valid] = ty_ids[valid].astype(int)
        ty_ids = ty_ids_int
    else:
        ty_ids = ty_ids.astype(int)

    n_time, n_sta = wind_speeds.shape

    if id_to_index:
        items = sorted([(tid, int(str(idx).strip())) for tid, idx in id_to_index.items() if str(idx).strip().isdigit()], key=lambda kv: kv[0])
    else:
        uniq = sorted({int(x) for x in np.unique(ty_ids) if int(x) >= 0})
        items = [(str(idx), idx) for idx in uniq]
    
    print(f"数据读取完毕。共 {n_sta} 个站点，{len(items)} 个台风过程。")

    # 3. 构建 (N台风, M站点) 特征矩阵 (不变)
    print(f"\n正在为 {len(items)} 个台风构建“{level_name}”空间分布特征向量...")
    
    feature_vectors = []
    typhoon_metadata = [] 
    
    for tid_str, ty_idx in items:
        typhoon_hours_vector = np.zeros(n_sta, dtype=float)
        
        for i in range(n_sta): 
            mask_ty = (ty_ids[:, i] == ty_idx)
            if not np.any(mask_ty):
                typhoon_hours_vector[i] = 0.0
                continue
            
            ws = wind_speeds[mask_ty, i]
            mask_wind = (ws >= thresh_min) & (ws <= thresh_max)
            hours = int(np.sum(mask_wind))
            typhoon_hours_vector[i] = hours
            
        if np.sum(typhoon_hours_vector) > 0:
            feature_vectors.append(typhoon_hours_vector)
            typhoon_metadata.append({
                "TID": tid_str,
                "Index": ty_idx,
                "CN_Name": index_to_cn.get(str(ty_idx), ""),
                "EN_Name": index_to_en.get(str(ty_idx), "")
            })
        else:
            print(f"  [INFO] 台风 {tid_str} ({index_to_cn.get(str(ty_idx), '')}) 无 {level_name} 影响, 已跳过。")

    # 4. 准备聚类
    X = np.array(feature_vectors)
    df_typhoons_meta = pd.DataFrame(typhoon_metadata)

    if X.shape[0] < MIN_CLUSTER_SIZE * 2: 
        print(f"[ERROR] 有效台风数 ({X.shape[0]}) 不足, 无法执行 HDBSCAN。")
        return

    print(f"\n特征矩阵构建完毕: (N_台风 = {X.shape[0]}, M_站点 = {X.shape[1]})")

    # --- 步骤 5: 使用 StandardScaler ---
    print("正在使用 StandardScaler 对特征矩阵进行标准化 (放大异常值)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 步骤 6: 使用 UMAP 降维 ---
    print(f"正在使用 UMAP 将数据从 {X_scaled.shape[1]} 维降至 {N_COMPONENTS_CLUSTER} 维...")
    
    umap_cluster_model = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS_CLUSTER,
        min_dist=MIN_DIST,
        metric='euclidean',
        random_state=42
    )
    X_umap_cluster = umap_cluster_model.fit_transform(X_scaled)
    print("UMAP D (用于聚类) 完成。")

    # --- 步骤 7: 使用 HDBSCAN 聚类 ---
    print(f"\n--- 正在 {N_COMPONENTS_CLUSTER} 维 UMAP 空间上执行 HDBSCAN ---")
    print(f"--- min_cluster_size = {MIN_CLUSTER_SIZE}, min_samples = {MIN_SAMPLES_PARAM} ---")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES_PARAM,  
        metric='euclidean',
        gen_min_span_tree=True 
    )
    
    clusterer.fit(X_umap_cluster)
    
    labels = clusterer.labels_ 
    
    # 8. 分析和保存结果
    df_typhoons_meta['Cluster'] = labels
    
    print("\n--- HDBSCAN 聚类结果统计 ---")
    cluster_counts = df_typhoons_meta['Cluster'].value_counts().sort_index()
    print(cluster_counts)
    
    n_clusters = len(cluster_counts[cluster_counts.index != -1])
    n_noise = cluster_counts.get(-1, 0)
    
    print(f"\n总结：")
    print(f"  > 找到 {n_clusters} 个主要簇")
    print(f"  > 识别到 {n_noise} 个台风为“噪声” (标签 -1)")
    
    if n_clusters >= 2:
        core_samples_mask = (labels != -1)
        core_X = X_umap_cluster[core_samples_mask]
        core_labels = labels[core_samples_mask]
        score = silhouette_score(core_X, core_labels)
        print(f"  > 核心簇的轮廓系数 (在 UMAP 空间): {score:.4f}")
    else:
        print("  > 核心簇不足2个，无法计算轮廓系数。")

    # --- 保存聚类分配表 ---
    assignments_csv_path = output_dir / f"Typhoon_Cluster_Assignments_HDBSCAN_{safe_level_name}.csv"
    df_typhoons_meta.to_csv(assignments_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 台风聚类归属表已保存: {assignments_csv_path.resolve()}")

    # --- 绘制每个【非噪声】簇的平均分布图 (逻辑不变) ---
    print(f"\n--- 正在为 {n_clusters} 个核心簇绘制平均分布图 ---")
    
    for cluster_id in cluster_counts.index:
        if cluster_id == -1: 
            continue
            
        cluster_typhoons = df_typhoons_meta[df_typhoons_meta['Cluster'] == cluster_id]
        indices_in_cluster = cluster_typhoons.index.values
        cluster_vectors = X[indices_in_cluster]
        avg_footprint = np.mean(cluster_vectors, axis=0)
        
        n_typhoons = len(cluster_typhoons)
        title = f"HDBSCAN 聚类 (ms={MIN_SAMPLES_PARAM}): {level_name} (簇={cluster_id})\n包含 {n_typhoons} 个台风的【平均】空间分布"
        
        fname = f"Typhoon_Cluster_HDBSCAN_{safe_level_name}_C{cluster_id}_AvgFootprint.png"
        png_path = output_dir / fname
        
        draw_station_count_text_map(
            lons, lats, avg_footprint, stids, title, str(png_path),
            threshold_val=thresh_min,
            extent=EXTENT, text_size=TEXT_SIZE, cmap_name=CMAP_NAME, show_zero=SHOW_ZERO
        )
        print(f"  [OK] 已保存地图: {fname}")

    # --- 步骤 9: 绘制 2D UMAP 可视化图 ---
    print("\n--- 正在生成 2D UMAP 可视化散点图 ---")
    
    umap_viz_model = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=2, 
        min_dist=0.3, 
        random_state=42
    )
    X_umap_viz = umap_viz_model.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('Spectral', len(unique_labels[unique_labels != -1]))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels[unique_labels != -1])}
    color_map[-1] = (0.7, 0.7, 0.7, 0.5) 
    
    for i, label in enumerate(labels):
        plt.scatter(
            X_umap_viz[i, 0], X_umap_viz[i, 1],
            color=color_map[label],
            s=50,
            alpha=0.8
        )
        plt.text(
            X_umap_viz[i, 0] + 0.01, X_umap_viz[i, 1] + 0.01,
            df_typhoons_meta.iloc[i]['TID'],
            fontsize=7
        )

    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[label], 
                          label=f"簇 {label} (N={cluster_counts.get(label, 0)})") 
               for label in cluster_counts.index]
    
    plt.legend(handles=handles, title="HDBSCAN 聚类结果")
    plt.title(f'台风空间分布的 UMAP 可视化 (2D)\n(按 HDBSCAN, ms={MIN_SAMPLES_PARAM} 结果着色)', fontsize=16)
    plt.xlabel("UMAP 维度 1", fontsize=12)
    plt.ylabel("UMAP 维度 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    viz_plot_path = output_dir / f"Typhoon_Cluster_UMAP_2D_Visualization.png"
    plt.savefig(viz_plot_path, dpi=180)
    plt.close()
    
    print(f"[OK] 2D UMAP 可视化地图已保存: {viz_plot_path.resolve()}")

    print(f"\n{'='*50}")
    print("--- 所有 UMAP + HDBSCAN 任务完成 ---")
    print(f"请检查输出目录: {output_dir.resolve()}")
    print("="*50)


if __name__ == "__main__":
    main()