# -*- coding: utf-8 -*-
"""
脚本名称：绘制站点数据质量分布图_业务缺测率版_Fixed.py
功能：
1. 读取 NC 文件获取经纬度。
2. 读取 CSV 获取 'Operational_Missing_Rate'。
3. 按照新的四档标准（优、良、中、差）绘制空间分布图。
4. 地形背景场设置严格参考了 '查看_站点质量.py'。
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import os
import sys

# ================= 1. 路径配置 (MacOS) =================

# 输入文件路径
BASE_DIR = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据/"
NC_FILE_PATH = os.path.join(BASE_DIR, "Combine_Stations_ExMaxWind_Fixed_2004_2024.nc")
CSV_FILE_PATH = os.path.join(BASE_DIR, "4_合并后_区分缺测类型_按站点.csv")

# 输出图片路径
OUTPUT_FIG_PATH = os.path.join(BASE_DIR, "2004_2024_合并后_业务缺测率分布图.png")

# SHP 文件路径 (参考您之前的代码设置，若Mac上没有请忽略，代码会自动处理)
# 建议：如果您有这些SHP文件，请修改下面的路径为实际路径
SHP_PATHS = {
    'zhejiang_province': '/Users/momo/Desktop/GIS_Data/zhejiang_province/Zhejiang_province.shp', 
    'zhejiang_city': '/Users/momo/Desktop/GIS_Data/zhejiang_city/Zhejiang_city.shp'
}

# ================= 2. 数据处理函数 =================

def load_and_merge_data():
    print(f"1. 正在读取 NetCDF 文件 (经纬度源): {NC_FILE_PATH}")
    try:
        with xr.open_dataset(NC_FILE_PATH) as ds:
            # 提取经纬度
            # 假设维度是 STID，变量是 lat/lon
            if 'lat' in ds and 'lon' in ds:
                lats = ds['lat'].values
                lons = ds['lon'].values
                stids = ds['STID'].values
            else:
                # 兼容某些NC文件 lat/lon 是坐标的情况
                stids = ds['STID'].values
                lats = ds.coords['lat'].values
                lons = ds.coords['lon'].values
            
            # 转为 DataFrame
            df_loc = pd.DataFrame({'STID': stids, 'Lat': lats, 'Lon': lons})
            df_loc['STID'] = df_loc['STID'].astype(str).str.strip()
            
    except Exception as e:
        print(f"  [!] 读取 NC 文件失败: {e}")
        return None

    print(f"2. 正在读取 CSV 文件 (质量源): {CSV_FILE_PATH}")
    try:
        df_csv = pd.read_csv(CSV_FILE_PATH)
        if 'STID' not in df_csv.columns:
            print("  [!] CSV 文件缺少 'STID' 列")
            return None
        if 'Operational_Missing_Rate' not in df_csv.columns:
            print("  [!] CSV 文件缺少 'Operational_Missing_Rate' 列")
            return None
            
        df_csv['STID'] = df_csv['STID'].astype(str).str.strip()
        
    except Exception as e:
        print(f"  [!] 读取 CSV 文件失败: {e}")
        return None

    print("3. 正在合并数据...")
    # 内连接合并
    df_merged = pd.merge(df_csv, df_loc, on='STID', how='inner')
    print(f"   合并后有效站点数: {len(df_merged)}")
    
    return df_merged

# ================= 3. 绘图主函数 =================

def plot_station_quality(df):
    # 设置字体 (MacOS 使用 Heiti TC 或 Arial Unicode MS)
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布
    fig = plt.figure(figsize=(12, 12), dpi=200)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # --- A. 地形背景场 (参考 '查看_站点质量.py' 设置) ---
    
    # 1. 设置范围 (参考原代码: 118-123, 27-31.5)
    extent = [118.0, 123.0, 27.0, 31.5]
    ax.set_extent(extent, crs=proj)

    print("4. 正在加载地形背景...")
    try:
        # 尝试加载 SHP 文件
        zj_shp = SHP_PATHS['zhejiang_province']
        city_shp = SHP_PATHS['zhejiang_city']
        
        has_shp = False
        if os.path.exists(zj_shp):
            zj_reader = shpreader.Reader(zj_shp)
            # 省界加粗
            ax.add_geometries(zj_reader.geometries(), crs=proj, 
                            edgecolor='black', facecolor='None', lw=1.2, zorder=1)
            has_shp = True
            
        if os.path.exists(city_shp):
            city_reader = shpreader.Reader(city_shp)
            # 市界变细，灰色
            ax.add_geometries(city_reader.geometries(), crs=proj, 
                            edgecolor='gray', facecolor='None', lw=0.5, linestyle='--', zorder=1)
            has_shp = True
            
        if has_shp:
            print("   [✓] 已加载本地 SHP 文件作为背景。")
        else:
            raise FileNotFoundError("本地 SHP 文件不存在")

    except Exception as e:
        print(f"   [!] 未找到本地 SHP 文件或加载失败 ({e})，切换至 Cartopy 默认背景。")
        # 备用方案：使用 Cartopy 自带数据
        provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none',
            edgecolor='black'
        )
        ax.add_feature(provinces, lw=0.8, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), lw=1.0, zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', lw=0.5, zorder=1)
        # 可选：添加河流湖泊增加细节
        ax.add_feature(cfeature.LAKES, alpha=0.3)

    # 网格线 (参考原代码风格)
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # --- B. 站点绘制 (按新标准分级) ---
    
    print("5. 正在绘制分类站点...")
    col_name = 'Operational_Missing_Rate'
    
    # 1. 优: 0% (完美) -> 蓝色
    df_opt = df[df[col_name] == 0]
    ax.scatter(df_opt['Lon'], df_opt['Lat'], s=15, c='dodgerblue', 
               label=f'优: 0% ({len(df_opt)}个)', alpha=0.8, transform=proj, zorder=2)

    # 2. 良: 0% < x < 5% -> 绿色
    df_good = df[(df[col_name] > 0) & (df[col_name] < 5)]
    ax.scatter(df_good['Lon'], df_good['Lat'], s=20, c='limegreen', 
               label=f'良: 0-5% ({len(df_good)}个)', alpha=0.8, transform=proj, zorder=3)

    # 3. 中: 5% <= x <= 10% -> 橙色
    df_fair = df[(df[col_name] >= 5) & (df[col_name] <= 10)]
    ax.scatter(df_fair['Lon'], df_fair['Lat'], s=40, c='orange', marker='D',
               label=f'中: 5-10% ({len(df_fair)}个)', alpha=0.9, edgecolors='white', lw=0.5, transform=proj, zorder=4)

    # 4. 差: > 10% -> 红色
    df_poor = df[df[col_name] > 10]
    ax.scatter(df_poor['Lon'], df_poor['Lat'], s=60, c='red', marker='X',
               label=f'差: >10% ({len(df_poor)}个)', alpha=1.0, edgecolors='black', lw=0.5, transform=proj, zorder=5)

    # --- C. 图例与保存 ---
    
    # 图例
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # 标题
    plt.title(f"台风大风代表站观测质量分布图\n总站数: {len(df)}", fontsize=15, pad=10)

    # 保存
    plt.savefig(OUTPUT_FIG_PATH, bbox_inches='tight', pad_inches=0.1)
    print(f"\n✅ 绘图完成! 图片已保存至: {OUTPUT_FIG_PATH}")
    # plt.show() # 在服务器或非交互环境下可注释此行

# ================= 4. 主程序入口 =================

if __name__ == "__main__":
    # 1. 加载合并数据
    df_data = load_and_merge_data()
    
    if df_data is not None:
        # 2. 执行绘图
        plot_station_quality(df_data)