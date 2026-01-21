import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
import cartopy.io.shapereader as shpreader
import warnings
import math  # [新增] 用于三角函数计算

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"
NC_FILE = os.path.join(BASE_DIR, "数据", "Refined_Combine_Stations_ExMaxWind_Fixed_2004_2024.nc")

# DEM 文件路径
DEM_FILE = os.path.join(BASE_DIR, "地形文件", "DEM_0P05_CHINA.nc") 
if not os.path.exists(DEM_FILE):
    DEM_FILE = os.path.join(BASE_DIR, "数据", "DEM_0P05_CHINA.nc")

OUTPUT_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Station_Static_Features_MultiScale.csv")

# [新增] 定义需要计算的半径列表 (单位: km)
TARGET_RADII = [5, 10, 15]

# ================= 核心计算函数 =================

def get_coastline_points(resolution='10m'):
    """
    获取高精度海岸线点集
    """
    print(f"正在加载 {resolution} 精度海岸线数据 (可能需要下载)...")
    try:
        shp_path = shpreader.natural_earth(resolution=resolution, category='physical', name='coastline')
        reader = shpreader.Reader(shp_path)
        coast_points = []
        
        # 筛选范围 (浙江及周边)
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
    """
    计算站点到最近海岸线的距离 (km)
    """
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
    """
    [修改] 支持多半径列表，并修正经度变形
    返回: 字典 {radius: complexity_array}
    """
    if not os.path.exists(dem_path):
        print(f"[跳过] 未找到 DEM 文件: {dem_path}")
        return None

    print(f"正在读取 DEM 数据: {dem_path} ...")
    try:
        nc = Dataset(dem_path)
        keys = nc.variables.keys()
        
        lat_name = next((k for k in keys if k.lower() in ['lat', 'latitude']), None)
        lon_name = next((k for k in keys if k.lower() in ['lon', 'longitude']), None)
        possible_names = ['hgt_m', 'hgt', 'hgt_era', 'z', 'height', 'elevation', 'data', 'band1']
        elev_name = next((k for k in keys if k.lower() in possible_names and k not in [lat_name, lon_name]), None)
        
        if not (lat_name and lon_name and elev_name):
            return None
            
        dem_lats = nc.variables[lat_name][:]
        dem_lons = nc.variables[lon_name][:]
        dem_elev = nc.variables[elev_name][:] 
        
        n_stations = len(station_lons)
        
        # 初始化结果字典，每个半径对应一个全0数组
        results = {r: np.zeros(n_stations) for r in radius_list}
        
        print(f"开始计算 {n_stations} 个站点的地形复杂度，涉及半径: {radius_list} km...")
        
        for i in range(n_stations):
            slat, slon = station_lats[i], station_lons[i]
            
            # 针对每个半径分别计算
            for r in radius_list:
                # [修正] 经度方向的度数计算考虑纬度修正
                # 1度纬度 ≈ 111km
                # 1度经度 ≈ 111 * cos(lat) km
                delta_lat = r / 111.0
                
                # 防止极点 cos(90)=0 的情况 (虽然不太可能出现在浙江)
                cos_lat = math.cos(math.radians(slat))
                if cos_lat < 0.01: cos_lat = 0.01 
                
                delta_lon = r / (111.0 * cos_lat)
                
                # 定义边界
                lat_min, lat_max = slat - delta_lat, slat + delta_lat
                lon_min, lon_max = slon - delta_lon, slon + delta_lon
                
                # 找到索引
                lat_indices = np.where((dem_lats >= lat_min) & (dem_lats <= lat_max))[0]
                lon_indices = np.where((dem_lons >= lon_min) & (dem_lons <= lon_max))[0]
                
                if len(lat_indices) == 0 or len(lon_indices) == 0:
                    results[r][i] = 0
                    continue
                    
                idx_lat_start, idx_lat_end = lat_indices.min(), lat_indices.max()
                idx_lon_start, idx_lon_end = lon_indices.min(), lon_indices.max()
                
                subset = dem_elev[idx_lat_start:idx_lat_end+1, idx_lon_start:idx_lon_end+1]
                
                if subset.size > 0:
                    val = np.std(subset)
                    if np.ma.is_masked(val):
                        results[r][i] = 0
                    else:
                        results[r][i] = val
                else:
                    results[r][i] = 0
        
        nc.close()
        return results

    except Exception as e:
        print(f"[错误] DEM 计算失败: {e}")
        return None

def calculate_terrain_complexity_neighbors(lons, lats, heights, radius_list=[5, 10, 15]):
    """
    (备选) 多半径支持
    """
    print(f"降级模式：计算基于站点的复杂度，半径: {radius_list} km...")
    n = len(lons)
    results = {r: np.zeros(n) for r in radius_list}
    
    points = np.column_stack((lons, lats))
    tree = cKDTree(points)
    
    for r in radius_list:
        # 这里为了简化，邻近点搜索暂用简单的度数近似 (1度~100km)，或者你也可用严格的 cos 修正
        # 但既然是降级方案，这里做个简单转换即可
        radius_deg = r / 100.0 
        indices_list = tree.query_ball_point(points, r=radius_deg)
        
        for i, neighbors in enumerate(indices_list):
            if len(neighbors) < 2:
                results[r][i] = 0.0
            else:
                neighbor_heights = heights[neighbors]
                results[r][i] = np.std(neighbor_heights)
            
    return results

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
    # 结果是一个字典: {5: array(...), 10: array(...), 15: array(...)}
    terrain_comp_dict = calculate_terrain_complexity_from_dem(lons, lats, DEM_FILE, radius_list=TARGET_RADII)
    
    if terrain_comp_dict is None:
        print("降级使用站点邻近法计算复杂度...")
        terrain_comp_dict = calculate_terrain_complexity_neighbors(lons, lats, heights, radius_list=TARGET_RADII)
    else:
        print("DEM 地形复杂度计算完成。")
    
    # 3. 整合数据
    data_dict = {
        'STID': stids,
        'Lat': lats,
        'Lon': lons,
        'Height': heights,
        'Dist_to_Coast': dist_coast
    }
    
    # 动态添加复杂度列
    for r in TARGET_RADII:
        col_name = f'Terrain_Complexity_{r}km'
        data_dict[col_name] = terrain_comp_dict[r]
        
    df_static = pd.DataFrame(data_dict)
    
    # 4. 保存
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_static.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\n[完成] 站点静态特征表已生成: {OUTPUT_CSV}")
    print(df_static.head())
    print(f"\n列说明:")
    print("- Dist_to_Coast: 距离最近海岸线的距离 (km)")
    for r in TARGET_RADII:
         print(f"- Terrain_Complexity_{r}km: 半径 {r}km 内的海拔标准差")

if __name__ == "__main__":
    main()