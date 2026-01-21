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
OUTPUT_CSV = os.path.join(BASE_DIR, "输出_机器学习", "Station_Static_Features.csv")

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

def calculate_terrain_complexity(lons, lats, heights, radius_km=10.0):
    """
    计算地形复杂度 (开阔度代理)
    方法：计算每个站点周围 R 公里范围内所有站点的海拔标准差
    StdDev 越大 -> 地形越崎岖 (山区)
    StdDev 越小 -> 地形越平坦 (平原/海面)
    """
    print(f"计算地形复杂度 (搜索半径 {radius_km} km)...")
    n = len(lons)
    complexity = np.zeros(n)
    
    # 粗略转换：在浙江纬度(约30度)，1度经度 ≈ 96km，1度纬度 ≈ 111km
    # 搜索半径 10km ≈ 0.1 度
    radius_deg = radius_km / 100.0 
    
    points = np.column_stack((lons, lats))
    tree = cKDTree(points)
    
    # query_ball_point 找到半径内的所有邻居
    indices_list = tree.query_ball_point(points, r=radius_deg)
    
    for i, neighbors in enumerate(indices_list):
        if len(neighbors) < 2:
            complexity[i] = 0.0 # 孤立点，复杂度设为0
        else:
            # 计算邻居的海拔标准差
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
    
    # 2. 计算地形复杂度
    terrain_comp = calculate_terrain_complexity(lons, lats, heights, radius_km=10.0)
    
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
    print("- Terrain_Complexity: 周围10km范围内海拔标准差 (值越大越崎岖)")

if __name__ == "__main__":
    main()