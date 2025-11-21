"""
大风分布空间约束聚类.py — 基于地理连通性的台风大风区域划分工具

【核心改进】：
1. 算法：使用 AgglomerativeClustering + kneighbors_graph 实现空间约束聚类。
2. 优化：修复了 Matplotlib 颜色警告。
3. 新增：在地图标题和文件名中自动标注轮廓系数，便于快速筛选最优结果。

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 引入聚类相关的库
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 引入地图绘制库
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# ==========================================
# 1. 全局配置 (Config)
# ==========================================

# 绘图字体设置
plt.rcParams['font.sans-serif'] = ['Heiti TC'] # Mac用 Heiti TC，Windows用 SimHei
plt.rcParams['axes.unicode_minus'] = False

# --- 文件路径配置 (保持您原有的路径) ---
CSV_EXCEED_PATH = "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/AllTyphoons_Exceed.csv"
CSV_EXACT_PATH  = "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/AllTyphoons_Exact.csv"
SHP_CITY_PATH   = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/shapefile/市界/浙江市界.shp"

# 输出根目录
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/大风累积空间聚类结果(空间约束版)") 

# --- 任务定义 ---
ANALYSIS_TASKS = [
    # --- 8级 (17.2 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_17.2",
        "name": "8级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_17.2",
        "name": "8级",
        "output_subdir": "指定级别 (Exact)"
    },
    
    # --- 9级 (20.8 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_20.8",
        "name": "9级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_20.8",
        "name": "9级",
        "output_subdir": "指定级别 (Exact)"
    },

    # --- 10级 (24.5 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_24.5",
        "name": "10级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_24.5",
        "name": "10级",
        "output_subdir": "指定级别 (Exact)"
    },
    
    # --- 11级 (28.5 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_28.5",
        "name": "11级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_28.5",
        "name": "11级",
        "output_subdir": "指定级别 (Exact)"
    },
    
    # --- 12级 (32.7 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_32.7",
        "name": "12级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_32.7",
        "name": "12级",
        "output_subdir": "指定级别 (Exact)"
    }
]

# 测试的分类数量 K 值范围
K_RANGE = range(2, 7) 

# 空间约束参数
N_NEIGHBORS = 15 

# ==========================================
# 2. 主逻辑 (Main Loop)
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"🚀 任务开始，总输出目录: {OUTPUT_DIR.resolve()}")

for task in ANALYSIS_TASKS:
    file_path = Path(task['file'])
    column = task['column']
    name = task['name']
    
    task_output_dir = OUTPUT_DIR / task['output_subdir']
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"正在处理任务: [{name}]")
    print(f"目标列: {column}")
    print(f"{'='*70}")
    
    try:
        # 1. 加载与数据准备
        df = pd.read_csv(file_path)
        if df.empty:
            continue
            
        features = df[['Lon', 'Lat', column]]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 2. 构建空间约束图
        print(f"🔗 正在构建空间约束图 (Neighbor k={N_NEIGHBORS})...")
        connectivity = kneighbors_graph(
            df[['Lon', 'Lat']], 
            n_neighbors=N_NEIGHBORS, 
            include_self=False
        )
        
        silhouette_list = []
        k_range_list = list(K_RANGE)
        
        # 3. 循环测试 K 值
        for k in k_range_list:
            print(f"  👉 正在尝试分区数 K={k} ...")
            
            # A. 聚类
            model = AgglomerativeClustering(
                n_clusters=k, 
                connectivity=connectivity, 
                linkage='ward'
            )
            labels = model.fit_predict(features_scaled)
            
            # B. 计算轮廓系数
            try:
                score = silhouette_score(features_scaled, labels)
            except ValueError:
                score = -1.0 # 异常情况
            
            silhouette_list.append(score)
            
            # C. 保存数据
            df_k = df.copy()
            df_k['Cluster'] = labels
            
            # 格式化分数用于文件名 (保留3位小数)
            score_str = f"{score:.3f}"
            
            # 保存 CSV (文件名也带上分数，方便对应)
            data_csv_path = task_output_dir / f"Clustered_Data_{name}_k{k}_Score{score_str}.csv"
            df_k.to_csv(data_csv_path, index=False, encoding='utf-8-sig') 
            
            # D. 绘制地图
            fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # 【修改点1】标题增加轮廓系数
            ax.set_title(f"空间约束聚类 (K={k}): {name} | 轮廓系数: {score_str}", fontsize=16)
            
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
            
            try:
                city_shapes = list(shpreader.Reader(SHP_CITY_PATH).geometries())
                ax.add_geometries(city_shapes, ccrs.PlateCarree(), 
                                  edgecolor='gray', facecolor='none', 
                                  linewidth=0.5, linestyle='--')
            except Exception:
                pass

            ax.set_extent([118, 123, 27, 31.5], crs=ccrs.PlateCarree())
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            # 【修改点2】使用新版颜色API，消除警告
            colors = matplotlib.colormaps['tab10']
            unique_labels = sorted(df_k['Cluster'].unique())
            
            for i, label in enumerate(unique_labels):
                cluster_data = df_k[df_k['Cluster'] == label]
                avg_hours = cluster_data[column].mean()
                
                ax.scatter(cluster_data['Lon'], cluster_data['Lat'], 
                           color=colors(i), 
                           label=f'区域 {label} (均值:{avg_hours:.1f}h)', 
                           s=20, 
                           transform=ccrs.PlateCarree(),
                           alpha=0.8, 
                           edgecolors='none')

            ax.legend(title="聚类区域", loc='upper right', fontsize=10)
            
            # 【修改点3】文件名最后增加轮廓系数
            map_png_path = task_output_dir / f"Clustered_Map_{name}_k{k}_Score{score_str}.png"
            
            fig.savefig(map_png_path, dpi=180, bbox_inches='tight')
            plt.close(fig)
            print(f"     [OK] 地图已保存 (Score={score_str}): {map_png_path.name}")

        # 保存指标汇总
        df_metrics = pd.DataFrame({
            'k': k_range_list,
            'Silhouette_Score': silhouette_list
        })
        metrics_csv_path = task_output_dir / f"K_Metrics_{name}.csv"
        df_metrics.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 任务完成，指标已保存。")

    except Exception as e:
        print(f"❌ [ERROR] 处理任务 {name} 时发生错误: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("🎉 全部处理完成！")