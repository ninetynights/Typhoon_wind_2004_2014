import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS'] # 适配不同系统的中文字体
plt.rcParams['axes.unicode_minus'] = False 

# ================= 1. 读取数据 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"
# 更新为包含 Slope 和 TPI 的新文件
csv_file = os.path.join(BASE_DIR, "输出_机器学习", 'Station_Static_Features_MultiScale_TPI.csv')

print(f"正在读取文件: {csv_file}")
df = pd.read_csv(csv_file)
# 清洗异常值 (例如海拔 -9999)
df_clean = df[df['Height'] > -100].copy()

# ================= 2. 可视化分析 =================

# 定义画布大小 (增加高度以容纳第三行)
fig = plt.figure(figsize=(24, 18))

# -----------------------------------------------------------
# 第一行：原有统计分析 (复杂度分布、热力图、尺度效应)
# -----------------------------------------------------------

# --- 子图 1: 地形复杂度分布直方图 ---
ax1 = fig.add_subplot(3, 3, 1)
sns.kdeplot(df_clean['Terrain_Complexity_5km'], color='skyblue', fill=True, label='5km', alpha=0.6, ax=ax1)
sns.kdeplot(df_clean['Terrain_Complexity_10km'], color='orange', fill=True, label='10km', alpha=0.4, ax=ax1)
sns.kdeplot(df_clean['Terrain_Complexity_15km'], color='green', fill=True, label='15km', alpha=0.3, ax=ax1)
ax1.set_title('分布对比: 地形复杂度 (5/10/15km)', fontsize=14)
ax1.set_xlabel('Terrain Complexity')
ax1.legend()

# --- 子图 2: 特征相关性热力图 (更新: 加入 Slope 和 TPI) ---
ax2 = fig.add_subplot(3, 3, 2)
# 定义要分析相关性的列
corr_cols = ['Height', 'Dist_to_Coast', 
             'Terrain_Complexity_5km', 'Terrain_Complexity_15km', 
             'Slope_Deg', 'TPI']
short_labels = ['Hgt', 'Dist', 'TC_5k', 'TC_15k', 'Slope', 'TPI']

# 计算相关系数
corr_matrix = df_clean[corr_cols].corr()

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
            xticklabels=short_labels, yticklabels=short_labels, ax=ax2)
ax2.set_title('全特征相关性热力图 (新增 Slope/TPI)', fontsize=14)

# --- 子图 3: 尺度效应 (原有) ---
ax3 = fig.add_subplot(3, 3, 3)
ax3.scatter(df_clean['Terrain_Complexity_5km'], df_clean['Terrain_Complexity_10km'], 
            alpha=0.5, s=15, c='orange', label='5km vs 10km')
ax3.scatter(df_clean['Terrain_Complexity_5km'], df_clean['Terrain_Complexity_15km'], 
            alpha=0.5, s=15, c='green', label='5km vs 15km')
max_val = df_clean['Terrain_Complexity_15km'].max()
ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')
ax3.set_title('尺度效应对比 (基准: 5km)', fontsize=14)
ax3.set_xlabel('Complexity (5km)')
ax3.legend()

# -----------------------------------------------------------
# 定义地图辅助函数 (复用)
# -----------------------------------------------------------
def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    provinces = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces_lines',
        scale='10m', facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linewidth=0.6, linestyle='--')
    # 锁定范围 (根据数据范围自动调整或固定为浙江周边)
    ax.set_extent([117.5, 123.5, 27, 31.5], crs=ccrs.PlateCarree())

# -----------------------------------------------------------
# 第二行：原有复杂度空间分布 (5/10/15km)
# -----------------------------------------------------------
vmin_tc = 0
vmax_tc = df_clean['Terrain_Complexity_15km'].quantile(0.98) #以此去除极值影响显示

# --- 子图 4 ---
ax4 = fig.add_subplot(3, 3, 4, projection=ccrs.PlateCarree())
add_map_features(ax4)
sc4 = ax4.scatter(df_clean['Lon'], df_clean['Lat'], c=df_clean['Terrain_Complexity_5km'], 
                  transform=ccrs.PlateCarree(), cmap='viridis', s=10, alpha=0.9, vmin=vmin_tc, vmax=vmax_tc)
ax4.set_title('空间分布: 5km 复杂度', fontsize=14)
plt.colorbar(sc4, ax=ax4, fraction=0.03, pad=0.04)

# --- 子图 5 ---
ax5 = fig.add_subplot(3, 3, 5, projection=ccrs.PlateCarree())
add_map_features(ax5)
sc5 = ax5.scatter(df_clean['Lon'], df_clean['Lat'], c=df_clean['Terrain_Complexity_10km'], 
                  transform=ccrs.PlateCarree(), cmap='viridis', s=10, alpha=0.9, vmin=vmin_tc, vmax=vmax_tc)
ax5.set_title('空间分布: 10km 复杂度', fontsize=14)
plt.colorbar(sc5, ax=ax5, fraction=0.03, pad=0.04)

# --- 子图 6 ---
ax6 = fig.add_subplot(3, 3, 6, projection=ccrs.PlateCarree())
add_map_features(ax6)
sc6 = ax6.scatter(df_clean['Lon'], df_clean['Lat'], c=df_clean['Terrain_Complexity_15km'], 
                  transform=ccrs.PlateCarree(), cmap='viridis', s=10, alpha=0.9, vmin=vmin_tc, vmax=vmax_tc)
ax6.set_title('空间分布: 15km 复杂度', fontsize=14)
plt.colorbar(sc6, ax=ax6, fraction=0.03, pad=0.04)

# -----------------------------------------------------------
# 第三行：[新增] Slope 和 TPI 的分析
# -----------------------------------------------------------

# --- 子图 7: Slope 和 TPI 的统计分布 ---
ax7 = fig.add_subplot(3, 3, 7)
# 使用双轴绘制两个不同量纲的分布
ax7_Right = ax7.twinx()

# 绘制 TPI (地形位置指数) - 蓝色填充
sns.kdeplot(df_clean['TPI'], ax=ax7, color='blue', fill=True, alpha=0.3, label='TPI (Left)')
ax7.set_ylabel('TPI Density', color='blue')
ax7.tick_params(axis='y', labelcolor='blue')
ax7.set_xlim(df_clean['TPI'].quantile(0.01), df_clean['TPI'].quantile(0.99)) # 去除极值优化显示

# 绘制 Slope (坡度) - 红色线条
sns.kdeplot(df_clean['Slope_Deg'], ax=ax7_Right, color='red', linestyle='--', linewidth=2, label='Slope (Right)')
ax7_Right.set_ylabel('Slope Density', color='red')
ax7_Right.tick_params(axis='y', labelcolor='red')

ax7.set_title('分布: 地形位置指数(TPI) 与 坡度(Slope)', fontsize=14)

# --- 子图 8: 空间分布 - Slope (坡度) ---
ax8 = fig.add_subplot(3, 3, 8, projection=ccrs.PlateCarree())
add_map_features(ax8)
# Slope 使用 Reds 色标，表示坡度陡峭程度
sc8 = ax8.scatter(df_clean['Lon'], df_clean['Lat'], c=df_clean['Slope_Deg'], 
                  transform=ccrs.PlateCarree(),
                  cmap='Reds', s=20, alpha=0.9, 
                  vmin=0, vmax=df_clean['Slope_Deg'].quantile(0.95)) # 截断前5%极值以增强对比度
ax8.set_title('空间分布: 坡度 (Slope_Deg)', fontsize=14)
cb8 = plt.colorbar(sc8, ax=ax8, fraction=0.03, pad=0.04)
cb8.set_label('Slope (°)')

# --- 子图 9: 空间分布 - TPI (地形位置指数) ---
ax9 = fig.add_subplot(3, 3, 9, projection=ccrs.PlateCarree())
add_map_features(ax9)
# TPI 使用 RdBu_r 色标 (红=正/山脊, 蓝=负/山谷, 白=0/平地)
# 使用 DivergingNorm 或 TwoSlopeNorm 确保 0 值对应白色
import matplotlib.colors as mcolors
div_norm = mcolors.TwoSlopeNorm(vmin=df_clean['TPI'].min(), vcenter=0., vmax=df_clean['TPI'].max())

sc9 = ax9.scatter(df_clean['Lon'], df_clean['Lat'], c=df_clean['TPI'], 
                  transform=ccrs.PlateCarree(),
                  cmap='RdBu_r', s=20, alpha=0.9, norm=div_norm)
ax9.set_title('空间分布: 地形位置指数 (TPI)', fontsize=14)
cb9 = plt.colorbar(sc9, ax=ax9, fraction=0.03, pad=0.04)
cb9.set_label('TPI (m)')

plt.tight_layout()
# 自动保存
output_img = os.path.join(BASE_DIR, "输出_机器学习", 'Station_Static_Features_Analysis.png')
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"\n[完成] 分析图已保存至: {output_img}")

plt.show()