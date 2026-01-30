"""
===============================================================================
脚本名称: 数据集构建_站点静态特征_DEM_v3.py
功能描述: 
    基于气象站点列表和高分辨率 DEM（数字高程模型）数据，计算并构建站点的
    静态地理环境特征集。该数据集旨在为机器学习模型（如台风大风预报）提供
    关键的环境背景信息。

核心算法:
    1. 离岸距离 (Dist_to_Coast): 
       利用 Cartopy 获取高精度海岸线，构建 KDTree 进行最近邻搜索，并计算 Haversine 球面距离。
    2. 地形复杂度 (Terrain Complexity / Roughness): 
       计算站点周围不同半径（5km, 10km, 15km）范围内海拔高度的标准差。
       反映了地形的崎岖程度，与风速的摩擦衰减密切相关。
    3. 坡度 (Slope): 
       基于 DEM 网格计算经向和纬向梯度，合成总坡度（角度）。
       反映了站点所在网格的宏观倾斜程度。
    4. 地形位置指数 (TPI, Topographic Position Index): 
       计算公式：TPI = 站点海拔 - 邻域平均海拔。
       物理意义：正值表示山脊/凸起，负值表示山谷/洼地，接近0表示平缓坡面或平原。
       使用 3x3 网格窗口（在 0.05° 分辨率下约对应 15km 尺度）。

输入 (Inputs):
    1. 站点元数据文件 (NetCDF):
       - 路径变量: NC_FILE
       - 必需变量: STID (站号), lat (纬度), lon (经度)
       - 可选变量: height (海拔)
    2. 数字高程模型 (NetCDF):
       - 路径变量: DEM_FILE (默认为 DEM_0P05_CHINA.nc)
       - 分辨率: 建议 0.05° 或更高
    3. 海岸线数据 (Shapefile):
       - 来源: Cartopy Natural Earth (首次运行需联网自动下载)

输出 (Outputs):
    1. 静态特征表 (CSV):
       - 路径变量: OUTPUT_CSV
       - 列结构:
         * STID, Lat, Lon, Height: 基础信息
         * Dist_to_Coast: 离岸距离 (km)
         * Terrain_Complexity_5km/10km/15km: 多尺度地形粗糙度
         * Slope_Deg: 坡度 (度)
         * TPI: 地形位置指数 (m)

注意事项 (Notes):
    1. 坐标系: 所有输入数据默认假设为 WGS84 坐标系。
    2. 缺失值处理: 若站点超出 DEM 覆盖范围，相关特征将填充为 0 或 NaN。
    3. 尺度效应: Slope 和 TPI 的计算强依赖于 DEM 的分辨率（当前代码基于 0.05°）。
       在 0.05° 尺度下，Slope 反映的是区域倾斜度而非微观坡度。
    4. 性能: 代码使用了全图矩阵运算优化 Slope/TPI 计算，效率远高于逐站循环。

依赖库:
    os, numpy, pandas, netCDF4, scipy, cartopy, math

===============================================================================
"""


import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter  # [新增] 用于计算 TPI 的平滑平均
import cartopy.io.shapereader as shpreader
import warnings
import math

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"
NC_FILE = os.path.join(BASE_DIR, "数据_v2", "Refined_Combine_Stations_ExMaxWind+SLP+StP_Fixed_2004_2024.nc")

# DEM 文件路径
DEM_FILE = os.path.join(BASE_DIR, "地形文件", "DEM_0P05_CHINA.nc") 
if not os.path.exists(DEM_FILE):
    DEM_FILE = os.path.join(BASE_DIR, "数据_v2", "DEM_0P05_CHINA.nc")

OUTPUT_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Station_Static_Features_MultiScale_TPI.csv")

# 定义地形复杂度的计算半径 (km)
TARGET_RADII = [5, 10, 15]

# ================= 核心计算函数 =================

def get_coastline_points(resolution='10m'):
    """获取高精度海岸线点集 (保持不变)"""
    print(f"正在加载 {resolution} 精度海岸线数据 (可能需要下载)...")
    try:
        shp_path = shpreader.natural_earth(resolution=resolution, category='physical', name='coastline')
        reader = shpreader.Reader(shp_path)
        coast_points = []
        bounds = (117, 126, 26, 33) 
        
        for geometry in reader.geometries():
            min_x, min_y, max_x, max_y = geometry.bounds
            if (min_x > bounds[1] or max_x < bounds[0] or 
                min_y > bounds[3] or max_y < bounds[2]):
                continue
            if geometry.geom_type == 'LineString':
                coast_points.extend(list(geometry.coords))
            elif geometry.geom_type == 'MultiLineString':
                for line in geometry.geoms:
                    coast_points.extend(list(line.coords))
                    
        if not coast_points:
            print("[警告] 未找到通过筛选的海岸线，将使用全球数据...")
            for geometry in reader.geometries():
                 if geometry.geom_type == 'LineString':
                    coast_points.extend(list(geometry.coords))
                 elif geometry.geom_type == 'MultiLineString':
                    for line in geometry.geoms:
                        coast_points.extend(list(line.coords))

        print(f"提取了 {len(coast_points)} 个海岸线节点用于距离计算。")
        return np.array(coast_points)
    except Exception as e:
        print(f"[错误] 加载海岸线失败: {e}")
        return None

def calculate_dist_to_coast(lons, lats):
    """计算离岸距离 (保持不变)"""
    coast_points = get_coastline_points()
    if coast_points is None:
        return np.zeros_like(lons)
    
    print("构建 KDTree 索引...")
    tree = cKDTree(coast_points)
    station_points = np.column_stack((lons, lats))
    
    print("查询最近海岸线点...")
    _, indices = tree.query(station_points, k=1)
    nearest_points = coast_points[indices]
    
    print("计算球面距离 (Haversine)...")
    R = 6371.0
    lon1, lat1 = np.radians(lons), np.radians(lats)
    lon2, lat2 = np.radians(nearest_points[:, 0]), np.radians(nearest_points[:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist_km = R * c
    return dist_km

def calculate_terrain_complexity_from_dem(station_lons, station_lats, dem_path, radius_list=[5, 10, 15]):
    """计算指定半径内的海拔标准差 (保持不变)"""
    if not os.path.exists(dem_path):
        return None

    print(f"正在读取 DEM 数据 (Complexity): {dem_path} ...")
    try:
        nc = Dataset(dem_path)
        keys = nc.variables.keys()
        
        lat_name = next((k for k in keys if k.lower() in ['lat', 'latitude']), None)
        lon_name = next((k for k in keys if k.lower() in ['lon', 'longitude']), None)
        possible_names = ['hgt_m', 'hgt', 'hgt_era', 'z', 'height', 'elevation', 'data', 'band1']
        elev_name = next((k for k in keys if k.lower() in possible_names and k not in [lat_name, lon_name]), None)
        
        if not (lat_name and lon_name and elev_name): return None
            
        dem_lats = nc.variables[lat_name][:]
        dem_lons = nc.variables[lon_name][:]
        dem_elev = nc.variables[elev_name][:] 
        
        n_stations = len(station_lons)
        results = {r: np.zeros(n_stations) for r in radius_list}
        
        print(f"计算 {n_stations} 个站点的地形粗糙度...")
        for i in range(n_stations):
            slat, slon = station_lats[i], station_lons[i]
            for r in radius_list:
                delta_lat = r / 111.0
                cos_lat = max(0.01, math.cos(math.radians(slat)))
                delta_lon = r / (111.0 * cos_lat)
                
                lat_indices = np.where((dem_lats >= slat - delta_lat) & (dem_lats <= slat + delta_lat))[0]
                lon_indices = np.where((dem_lons >= slon - delta_lon) & (dem_lons <= slon + delta_lon))[0]
                
                if len(lat_indices) == 0 or len(lon_indices) == 0:
                    results[r][i] = 0
                    continue
                
                subset = dem_elev[lat_indices.min():lat_indices.max()+1, lon_indices.min():lon_indices.max()+1]
                if subset.size > 0:
                    val = np.std(subset)
                    results[r][i] = 0 if np.ma.is_masked(val) else val
                else:
                    results[r][i] = 0
        
        nc.close()
        return results
    except Exception as e:
        print(f"[错误] Complexity 计算失败: {e}")
        return None

def calculate_slope_tpi_from_dem(station_lons, station_lats, dem_path):
    """
    [新增] 计算站点对应的 Slope (坡度) 和 TPI (地形位置指数)
    方法: 先计算全图的 Slope/TPI 矩阵，再提取站点值
    """
    if not os.path.exists(dem_path):
        print("[跳过] DEM 不存在，无法计算 Slope/TPI")
        return None, None

    print(f"正在读取 DEM 数据 (Slope/TPI): {dem_path} ...")
    try:
        nc = Dataset(dem_path)
        keys = nc.variables.keys()
        
        # 1. 自动识别变量名
        lat_name = next((k for k in keys if k.lower() in ['lat', 'latitude']), None)
        lon_name = next((k for k in keys if k.lower() in ['lon', 'longitude']), None)
        possible_names = ['hgt_m', 'hgt', 'hgt_era', 'z', 'height', 'elevation', 'data', 'band1']
        elev_name = next((k for k in keys if k.lower() in possible_names and k not in [lat_name, lon_name]), None)

        # 加载数据到内存
        dem_lats = nc.variables[lat_name][:]
        dem_lons = nc.variables[lon_name][:]
        dem_elev = nc.variables[elev_name][:]
        
        # 填充掩码值 (如果有)
        if np.ma.is_masked(dem_elev):
            dem_elev = dem_elev.filled(0)

        print("正在全图计算 Slope (坡度) ...")
        # --- 计算 Slope ---
        # np.gradient 返回 (axis0_grad, axis1_grad) 即 (y方向, x方向)
        grad_y, grad_x = np.gradient(dem_elev)
        
        # 假设 DEM 分辨率为 0.05 度
        res = 0.05 
        dist_y = res * 111320.0  # 纬度方向距离 (米)
        
        # 经度方向距离随纬度变化 (米)
        # 扩展 lat 维度以匹配 grid 形状 (N, 1) * (N, M)
        lat_grid = dem_lats[:, np.newaxis] 
        dist_x = res * 111320.0 * np.cos(np.radians(lat_grid))
        dist_x[dist_x < 1] = 1 # 避免除零

        # 计算坡度 (角度)
        slope_rad = np.arctan(np.sqrt((grad_y / dist_y)**2 + (grad_x / dist_x)**2))
        slope_grid = np.degrees(slope_rad)

        print("正在全图计算 TPI (地形位置指数) ...")
        # --- 计算 TPI ---
        # TPI = 海拔 - 邻域平均海拔
        # 窗口大小 size=3 (3x3 grid)，对应 0.05*3 = 0.15度 ≈ 15km 尺度
        mean_elev = uniform_filter(dem_elev, size=2, mode='reflect')
        tpi_grid = dem_elev - mean_elev
        
        # --- 提取站点值 ---
        print("正在提取站点特征值...")
        n_stations = len(station_lons)
        res_slope = np.zeros(n_stations)
        res_tpi = np.zeros(n_stations)
        
        # 简单的最近邻查找
        for i in range(n_stations):
            slat, slon = station_lats[i], station_lons[i]
            
            # 找到最近的索引
            lat_idx = (np.abs(dem_lats - slat)).argmin()
            lon_idx = (np.abs(dem_lons - slon)).argmin()
            
            res_slope[i] = slope_grid[lat_idx, lon_idx]
            res_tpi[i] = tpi_grid[lat_idx, lon_idx]

        nc.close()
        return res_slope, res_tpi

    except Exception as e:
        print(f"[错误] Slope/TPI 计算失败: {e}")
        return np.zeros(len(station_lons)), np.zeros(len(station_lons))


# ================= 主流程 =================

def main():
    print(f"读取 NC 文件: {NC_FILE}")
    if not os.path.exists(NC_FILE):
        print("[错误] 文件不存在")
        return
        
    nc = Dataset(NC_FILE)
    stids = np.array(nc.variables['STID'][:]).astype(str)
    lats = np.array(nc.variables['lat'][:])
    lons = np.array(nc.variables['lon'][:])
    
    if 'height' in nc.variables:
        heights = np.array(nc.variables['height'][:])
    else:
        heights = np.zeros_like(lats)
    nc.close()
    
    print(f"共 {len(stids)} 个站点。")
    
    # 1. 计算离岸距离
    dist_coast = calculate_dist_to_coast(lons, lats)
    
    # 2. 计算地形复杂度 (多尺度)
    terrain_comp_dict = calculate_terrain_complexity_from_dem(lons, lats, DEM_FILE, radius_list=TARGET_RADII)
    
    # 3. [新增] 计算 Slope 和 TPI
    slope_vals, tpi_vals = calculate_slope_tpi_from_dem(lons, lats, DEM_FILE)
    
    # 4. 整合数据
    data_dict = {
        'STID': stids,
        'Lat': lats,
        'Lon': lons,
        'Height': heights,
        'Dist_to_Coast': dist_coast
    }
    
    # 添加复杂度列
    for r in TARGET_RADII:
        col_name = f'Terrain_Complexity_{r}km'
        data_dict[col_name] = terrain_comp_dict[r]

    # 添加 Slope 和 TPI 列
    if slope_vals is not None:
        data_dict['Slope_Deg'] = slope_vals
        data_dict['TPI'] = tpi_vals
    else:
        # 如果计算失败，填0或NaN
        data_dict['Slope_Deg'] = np.zeros(len(stids))
        data_dict['TPI'] = np.zeros(len(stids))
        
    df_static = pd.DataFrame(data_dict)
    
    # 5. 保存
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_static.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\n[完成] 站点静态特征表已生成: {OUTPUT_CSV}")
    print(df_static.head())
    print(f"\n列说明:")
    print("- Dist_to_Coast: 距离最近海岸线的距离 (km)")
    print("- Slope_Deg: 站点所在网格的坡度 (度)")
    print("- TPI: 地形位置指数 (正值为山顶/凸起, 负值为山谷/洼地)")
    for r in TARGET_RADII:
         print(f"- Terrain_Complexity_{r}km: 半径 {r}km 内的海拔标准差")

if __name__ == "__main__":
    main()