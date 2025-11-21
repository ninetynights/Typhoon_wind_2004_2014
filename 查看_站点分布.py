import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import os

def plot_station_distribution(nc_file, csv_sets, output_png, shp_paths):
    """
    绘制站点分布图，按类别区分颜色。
    """
    print("开始绘制站点分布图...")
    
    # --- 1. 加载数据 ---
    print(f"  正在读取 NC 文件: {nc_file}")
    try:
        with xr.open_dataset(nc_file) as ds:
            # 提取所有站点的经纬度信息
            df_meta = ds[['lat', 'lon']].to_dataframe()
            # <--- 修正点 1 (安全措施) ---
            # 确保 NetCDF 的索引也是字符串，以防万一
            df_meta.index = df_meta.index.astype(str)
            # <--- 修正结束 ---
    except FileNotFoundError:
        print(f"  [!!] 错误: 找不到 NetCDF 文件: {nc_file}")
        return
    
    print(f"  正在读取 CSV 文件: {csv_sets}")
    try:
        # <--- 修正点 2 (核心修复) ---
        # 强制 Pandas 将所有列都读取为字符串(str)，防止它自动转为浮点数
        df_sets = pd.read_csv(csv_sets, dtype=str)
        # <--- 修正结束 ---
    except FileNotFoundError:
        print(f"  [!!] 错误: 找不到 CSV 文件: {csv_sets}")
        return

    # --- 2. 按类别分离站点 ---
    # .dropna() 确保我们只获取有效的站点ID
    common_stids = df_sets['common_stations'].dropna().tolist()
    national_stids = df_sets['national_only_stations'].dropna().tolist()
    township_stids = df_sets['township_only_stations'].dropna().tolist()

    # 现在 common_stids 是 ['58443', '58444', ...] (字符串)
    # df_meta.index 也是 ['58443', '58444', ...] (字符串)
    # .loc 现在可以正常工作了
    meta_common = df_meta.loc[common_stids]
    meta_national = df_meta.loc[national_stids]
    meta_township = df_meta.loc[township_stids]

    print(f"  站点分类完毕: 重叠({len(meta_common)}), 国家站独有({len(meta_national)}), 代表站独有({len(meta_township)})")

    # --- 3. 开始绘图 (参考你的样式) ---
    plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()  # 投影
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # 设置地图范围 (来自你的参考代码)
    lon_min, lon_max = 118, 123
    lat_min, lat_max = 27, 31.5
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # --- 4. 添加地理底图 ---
    print("  正在加载地理底图...")
    try:
        # --- 优先尝试加载你指定的 SHP 文件 ---
        zj_shp = shp_paths['zhejiang_province']
        city_shp = shp_paths['zhejiang_city']
        
        if not os.path.exists(zj_shp) or not os.path.exists(city_shp):
            raise FileNotFoundError("SHP 文件路径无效")

        zj_reader = shpreader.Reader(zj_shp)
        ax.add_geometries(zj_reader.geometries(), crs=proj, edgecolor='black', facecolor='None', lw=1.5)
        
        city_reader = shpreader.Reader(city_shp)
        ax.add_geometries(city_reader.geometries(), crs=proj, edgecolor='gray', facecolor='None', lw=0.5)
        
        print(f"  [✓] 成功加载本地 SHP 文件。")

    except Exception as e:
        # --- 备用方案 ---
        print(f"  [!] 警告: 加载本地 SHP 文件失败 ({e})。")
        print("         将使用 Cartopy 默认的省界和海岸线作为备用底图。")
        
        provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces',
            scale='10m',
            facecolor='none',
            edgecolor='black'
        )
        ax.add_feature(provinces, lw=0.8)
        ax.add_feature(cfeature.COASTLINE, lw=1.0)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

    # 绘制网格线 (来自你的参考代码)
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # --- 5. 绘制三类站点 ---
    print("  正在绘制站点...")
    
    # 代表站独有 (绿色) - 先画，放在最底层
    ax.scatter(meta_township['lon'], meta_township['lat'],
               s=10,  # 点的大小
               color='green',
               label=f'代表站独有 ({len(meta_township)})',
               alpha=0.7, 
               transform=proj)

    # 国家站独有 (蓝色)
    ax.scatter(meta_national['lon'], meta_national['lat'],
               s=10,
               color='blue',
               label=f'国家站独有 ({len(meta_national)})',
               alpha=0.7,
               transform=proj)

    # 重叠站点 (红色) - 最后画，放在最顶层
    ax.scatter(meta_common['lon'], meta_common['lat'],
               s=10,
               color='red',
               label=f'重叠站点 ({len(meta_common)})',
               alpha=0.7,
               transform=proj)

    # --- 6. 添加图例和标题并保存 ---
    ax.legend(loc='upper right', markerscale=2, fontsize=12)
    ax.set_title('国家站与代表站站点分布图', fontsize=16)

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✅ 成功! 站点分布图已保存到:\n   {output_png}")
    # plt.show() 


# --- 脚本主程序入口 ---
if __name__ == "__main__":
    
    # --- 1. 文件路径设置 ---
    base_dir = "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/"
    
    # 输入文件
    nc_file = os.path.join(base_dir, "Refined_Combine_Stations_ExMaxWind.nc")
    
    # <--- 修正点 3 (修正路径) ---
    # 根据你的报错日志，CSV文件在 "对比报告" 子目录中
    csv_sets = os.path.join(base_dir, "nc对比报告", "1_站点集合分析.csv")
    # <--- 修正结束 ---
    
    # 输出文件
    output_png = os.path.join(base_dir, "站点分布.png")
    
    # SHP 路径 (来自你的参考文件)
    # **注意**: 'N:' 路径在 macOS 上很可能无效。
    # 请确保这些路径对你当前的系统是正确的，否则脚本将自动使用备用底图。
    shp_paths = {
        'zhejiang_province': 'N:/00_GIS/shp/zhejiang/zhejiang_province/Zhejiang_province.shp',
        'zhejiang_city': 'N:/00_GIS/shp/zhejiang/zhejiang_city/Zhejiang_city.shp'
    }

    # --- 2. 运行绘图函数 ---
    plot_station_distribution(nc_file, csv_sets, output_png, shp_paths)