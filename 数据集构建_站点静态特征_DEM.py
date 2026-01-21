import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
import cartopy.io.shapereader as shpreader
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"
NC_FILE = os.path.join(BASE_DIR, "数据", "Refined_Combine_Stations_ExMaxWind_Fixed_2004_2024.nc")

# [新增] DEM 文件路径
# 假设它在 '地形文件' 或 '数据' 目录下，请根据实际情况修改
DEM_FILE = os.path.join(BASE_DIR, "地形文件", "DEM_0P05_CHINA.nc") 
if not os.path.exists(DEM_FILE):
    # 尝试在数据目录下找
    DEM_FILE = os.path.join(BASE_DIR, "数据", "DEM_0P05_CHINA.nc")

OUTPUT_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Station_Static_Features_DEM.csv")

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
        
        # 仅筛选中国及其周边的海岸线以加速计算 (粗略范围: 118-125E, 25-35N)
        # 浙江范围大致在 118E-123E, 27N-31N
        bounds = (117, 126, 26, 33) 
        
        for geometry in reader.geometries():
            # 简单的包围盒检查
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
            # 回退：读取所有
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
    使用 KDTree 加速最近邻搜索
    """
    coast_points = get_coastline_points()
    if coast_points is None:
        return np.zeros_like(lons)
    
    # 构建 KDTree
    print("构建 KDTree 索引...")
    # 注意：这里我们做平面投影近似计算，或者直接用经纬度寻找最近点
    # 为了精确，找到最近点的索引后，再算球面距离
    tree = cKDTree(coast_points)
    
    station_points = np.column_stack((lons, lats))
    
    print("查询最近海岸线点...")
    dists_deg, indices = tree.query(station_points, k=1)
    
    # 获取最近点的坐标
    nearest_points = coast_points[indices]
    
    # 计算精确的 Haversine 距离
    print("计算球面距离...")
    R = 6371.0
    lon1, lat1 = np.radians(lons), np.radians(lats)
    lon2, lat2 = np.radians(nearest_points[:, 0]), np.radians(nearest_points[:, 1])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist_km = R * c
    
    return dist_km

def calculate_terrain_complexity_from_dem(station_lons, station_lats, dem_path, radius_km=15.0):
    """
    [新增] 使用 DEM 数据计算地形复杂度
    原理：以站点为中心，截取 radius_km 范围内的 DEM 网格，计算海拔标准差
    """
    if not os.path.exists(dem_path):
        print(f"[跳过] 未找到 DEM 文件: {dem_path}")
        return None

    print(f"正在读取 DEM 数据: {dem_path} ...")
    try:
        nc = Dataset(dem_path)
        
        # 1. 自动寻找变量名 (处理不同的命名习惯)
        keys = nc.variables.keys()
        
        # 找纬度变量
        lat_name = next((k for k in keys if k.lower() in ['lat', 'latitude']), None)
        # 找经度变量
        lon_name = next((k for k in keys if k.lower() in ['lon', 'longitude']), None)
        
        # 找高度变量 (排除经纬度变量)
        # [修改] 增加了 hgt_m, hgt, hgt_era 等常见的 DEM 变量名
        possible_names = ['hgt_m', 'hgt', 'hgt_era', 'z', 'height', 'elevation', 'data', 'band1']
        elev_name = next((k for k in keys if k.lower() in possible_names 
                          and k not in [lat_name, lon_name]), None)
        
        if not (lat_name and lon_name and elev_name):
            print(f"[错误] 无法在 DEM 中识别经纬度或高度变量。Keys: {keys}")
            return None
            
        print(f"识别到 DEM 变量 -> Lat: {lat_name}, Lon: {lon_name}, Elev: {elev_name}")
        
        dem_lats = nc.variables[lat_name][:]
        dem_lons = nc.variables[lon_name][:]
        dem_elev = nc.variables[elev_name][:] # 假设是 2D 数组 [lat, lon]
        
        # 检查维度，如果是 1D 坐标轴 (常见的 grid 格式)
        if dem_lats.ndim == 1 and dem_lons.ndim == 1:
            # 这种是最标准的 (Lat, Lon) 网格
            pass
        else:
            print("[警告] DEM 经纬度不是 1D 向量，暂时不支持复杂投影。")
            return None
            
        print(f"DEM 加载成功。Grid Size: {dem_elev.shape}")
        
        # 2. 逐站点计算
        n_stations = len(station_lons)
        complexity = np.zeros(n_stations)
        
        # 粗略转换：半径转经纬度跨度
        # 1度 ≈ 111km
        delta_deg = radius_km / 100.0
        
        print(f"开始计算 {n_stations} 个站点的地形复杂度 (半径 ~{radius_km}km)...")
        
        for i in range(n_stations):
            slat, slon = station_lats[i], station_lons[i]
            
            # 定义边界 (Bounding Box)
            lat_min, lat_max = slat - delta_deg, slat + delta_deg
            lon_min, lon_max = slon - delta_deg, slon + delta_deg
            
            # 找到 DEM 中的索引范围
            # np.where 返回满足条件的索引
            lat_indices = np.where((dem_lats >= lat_min) & (dem_lats <= lat_max))[0]
            lon_indices = np.where((dem_lons >= lon_min) & (dem_lons <= lon_max))[0]
            
            if len(lat_indices) == 0 or len(lon_indices) == 0:
                # 站点在 DEM 范围外
                complexity[i] = 0
                continue
                
            # 提取切片
            # 注意：切片需要处理 Lat 是否是降序排列的情况
            idx_lat_start, idx_lat_end = lat_indices.min(), lat_indices.max()
            idx_lon_start, idx_lon_end = lon_indices.min(), lon_indices.max()
            
            # [Lat, Lon]
            subset = dem_elev[idx_lat_start:idx_lat_end+1, idx_lon_start:idx_lon_end+1]
            
            # 计算标准差 (忽略 NaN)
            if subset.size > 0:
                # 某些 DEM 海洋部分是 Masked 或 NaN
                val = np.std(subset)
                if np.ma.is_masked(val):
                    complexity[i] = 0
                else:
                    complexity[i] = val
            else:
                complexity[i] = 0
                
        nc.close()
        return complexity

    except Exception as e:
        print(f"[错误] DEM 计算失败: {e}")
        return None

def calculate_terrain_complexity_neighbors(lons, lats, heights, radius_km=10.0):
    """
    (备选方案) 计算每个站点周围 R 公里范围内所有站点的海拔标准差
    """
    print(f"计算基于站点的复杂度 (搜索半径 {radius_km} km)...")
    n = len(lons)
    complexity = np.zeros(n)
    
    radius_deg = radius_km / 100.0 
    
    points = np.column_stack((lons, lats))
    tree = cKDTree(points)
    
    indices_list = tree.query_ball_point(points, r=radius_deg)
    
    for i, neighbors in enumerate(indices_list):
        if len(neighbors) < 2:
            complexity[i] = 0.0
        else:
            neighbor_heights = heights[neighbors]
            complexity[i] = np.std(neighbor_heights)
            
    return complexity

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
        print("[警告] 未找到 height 变量，全设为 0")
        heights = np.zeros_like(lats)
        
    nc.close()
    
    print(f"共 {len(stids)} 个站点。")
    
    # 1. 计算离岸距离
    dist_coast = calculate_dist_to_coast(lons, lats)
    
    # 2. 计算地形复杂度 (优先尝试 DEM)
    # 尝试半径 15km (0.05度 DEM 约 5km 格距，15km 半径能覆盖约 5x5=25 个网格，统计意义较好)
    terrain_comp = calculate_terrain_complexity_from_dem(lons, lats, DEM_FILE, radius_km=15.0)
    
    if terrain_comp is None:
        print("降级使用站点邻近法计算复杂度...")
        terrain_comp = calculate_terrain_complexity_neighbors(lons, lats, heights, radius_km=10.0)
    else:
        print("DEM 地形复杂度计算完成。")
    
    # 3. 整合
    df_static = pd.DataFrame({
        'STID': stids,
        'Lat': lats,
        'Lon': lons,
        'Height': heights,
        'Dist_to_Coast': dist_coast,
        'Terrain_Complexity': terrain_comp
    })
    
    # 4. 保存
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_static.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\n[完成] 站点静态特征表已生成: {OUTPUT_CSV}")
    print(df_static.head())
    print(f"\n列说明:")
    print("- Dist_to_Coast: 距离最近海岸线的距离 (km)")
    print("- Terrain_Complexity: 地形崎岖度/开阔度代理 (DEM 高程标准差)")

if __name__ == "__main__":
    main()