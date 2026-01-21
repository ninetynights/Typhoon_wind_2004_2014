import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Heiti TC'] 
plt.rcParams['axes.unicode_minus'] = False 

# ================= 1. 读取数据 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"
csv_file = os.path.join(BASE_DIR, "输出_机器学习", 'Station_Static_Features_MultiScale.csv')

if not os.path.exists(csv_file):
    # 调试用 fallback
    csv_file = 'Station_Static_Features_MultiScale.csv'

print(f"正在读取文件: {csv_file}")
df = pd.read_csv(csv_file)
df_clean = df[df['Height'] > -100].copy()

# ================= 2. 可视化分析 =================

# 定义画布大小
fig = plt.figure(figsize=(24, 12))

# -----------------------------------------------------------
# 第一行：统计分析 (普通坐标轴)
# -----------------------------------------------------------

# --- 子图 1: 分布直方图 ---
ax1 = fig.add_subplot(2, 3, 1)
sns.kdeplot(df['Terrain_Complexity_5km'], color='skyblue', fill=True, label='5km', alpha=0.6, ax=ax1)
sns.kdeplot(df['Terrain_Complexity_10km'], color='orange', fill=True, label='10km', alpha=0.4, ax=ax1)
sns.kdeplot(df['Terrain_Complexity_15km'], color='green', fill=True, label='15km', alpha=0.3, ax=ax1)
ax1.set_title('分布对比: 地形复杂度 (5/10/15km)', fontsize=14)
ax1.set_xlabel('Terrain Complexity')
ax1.legend()

# --- 子图 2: 热力图 ---
ax2 = fig.add_subplot(2, 3, 2)
corr_cols = ['Height', 'Dist_to_Coast', 'Terrain_Complexity_5km', 'Terrain_Complexity_10km', 'Terrain_Complexity_15km']
short_labels = ['Hgt', 'Dist', 'TC_5k', 'TC_10k', 'TC_15k']
corr_matrix = df_clean[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
            xticklabels=short_labels, yticklabels=short_labels, ax=ax2)
ax2.set_title('特征相关性热力图', fontsize=14)

# --- 子图 3: 尺度效应 ---
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(df['Terrain_Complexity_5km'], df['Terrain_Complexity_10km'], 
            alpha=0.5, s=15, c='orange', label='5km vs 10km')
ax3.scatter(df['Terrain_Complexity_5km'], df['Terrain_Complexity_15km'], 
            alpha=0.5, s=15, c='green', label='5km vs 15km')
max_val = df['Terrain_Complexity_15km'].max()
ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')
ax3.set_title('尺度效应对比 (基准: 5km)', fontsize=14)
ax3.set_xlabel('Complexity (5km)')
ax3.set_ylabel('Complexity (10km & 15km)')
ax3.legend()

# -----------------------------------------------------------
# 第二行：空间分布 (地图投影坐标轴)
# -----------------------------------------------------------

# 定义绘图辅助函数：添加地图底图
def add_map_features(ax):
    # 1. 基础要素
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5) # 国界
    
    # 2. 添加省界 (使用 Natural Earth 数据)
    # 这里的 name='admin_1_states_provinces_lines' 包含了全球省界
    provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linewidth=0.6, linestyle='--')
    
    # 3. 锁定范围 (浙江及周边: 117E-124E, 26N-32N)
    ax.set_extent([117.5, 123.5, 27, 31.5], crs=ccrs.PlateCarree())
    
    # 4. 网格线 (可选)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

# 统一色标范围
vmin = 0
vmax = df['Terrain_Complexity_15km'].max() * 0.9 

# --- 子图 4: 空间分布 5km ---
ax4 = fig.add_subplot(2, 3, 4, projection=ccrs.PlateCarree())
add_map_features(ax4)
sc4 = ax4.scatter(df['Lon'], df['Lat'], c=df['Terrain_Complexity_5km'], 
                  transform=ccrs.PlateCarree(), # 关键：告诉 cartopy 散点数据是经纬度
                  cmap='viridis', s=15, alpha=0.9, vmin=vmin, vmax=vmax, edgecolors='none')
ax4.set_title('空间分布: 5km 复杂度', fontsize=14)

# --- 子图 5: 空间分布 10km ---
ax5 = fig.add_subplot(2, 3, 5, projection=ccrs.PlateCarree())
add_map_features(ax5)
sc5 = ax5.scatter(df['Lon'], df['Lat'], c=df['Terrain_Complexity_10km'], 
                  transform=ccrs.PlateCarree(),
                  cmap='viridis', s=15, alpha=0.9, vmin=vmin, vmax=vmax, edgecolors='none')
ax5.set_title('空间分布: 10km 复杂度', fontsize=14)

# --- 子图 6: 空间分布 15km ---
ax6 = fig.add_subplot(2, 3, 6, projection=ccrs.PlateCarree())
add_map_features(ax6)
sc6 = ax6.scatter(df['Lon'], df['Lat'], c=df['Terrain_Complexity_15km'], 
                  transform=ccrs.PlateCarree(),
                  cmap='viridis', s=15, alpha=0.9, vmin=vmin, vmax=vmax, edgecolors='none')
ax6.set_title('空间分布: 15km 复杂度', fontsize=14)

# 添加共用的 Colorbar
plt.tight_layout(rect=[0, 0, 0.92, 1]) 
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.3]) 
plt.colorbar(sc6, cax=cbar_ax, label='Terrain Complexity')

# 保存图片
# output_img = 'terrain_analysis_multiscale_map.png'
# plt.savefig(output_img, dpi=300, bbox_inches='tight')
# print(f"\n[完成] 带地图背景的分析图已保存为: {output_img}")

plt.show()