import xarray as xr
import pandas as pd
import os

def filter_nc_by_quality(original_nc_path, csv_quality_report, output_nc_path, keep_threshold=10.0):
    """
    根据质量报告筛选 NetCDF 文件中的站点。

    Args:
        original_nc_path (str): 原始 NetCDF 文件路径。
        csv_quality_report (str): 站点质量报告 CSV 文件路径。
        output_nc_path (str): 筛选后输出的 NetCDF 文件路径。
        keep_threshold (float): 保留的站点最大缺测率阈值 (例如 10.0)。
    """
    print("开始基于数据质量筛选 NetCDF 文件...")

    # --- 1. 读取质量报告并筛选“好站点” ---
    print(f"  正在读取 CSV 质量报告: {csv_quality_report}")
    try:
        # 强制将 STID 读作字符串，以匹配 NC 文件的索引
        df_quality = pd.read_csv(csv_quality_report, dtype={'STID': str})
    except FileNotFoundError:
        print(f"  [!!] 错误: 找不到 CSV 文件: {csv_quality_report}")
        return

    # 筛选出缺测率 <= 阈值 的站点
    df_good_stations = df_quality[df_quality['Operational_Missing_Rate'] <= keep_threshold]
    
    # 获取“好站点”的 ID 列表
    good_station_ids = df_good_stations['STID'].tolist()
    
    if not good_station_ids:
        print(f"  [!!] 错误: 在阈值 {keep_threshold}% 下没有找到任何符合条件的站点。")
        return

    print(f"  [i] 筛选条件: 缺测率 <= {keep_threshold}%")
    print(f"  [i] 发现 {len(good_station_ids)} 个高质量站点将被保留。")

    # --- 2. 加载原始 NC 文件并执行筛选 ---
    print(f"  正在加载原始 NC 文件: {original_nc_path}")
    num_original = 0
    num_filtered = 0
    
    try:
        with xr.open_dataset(original_nc_path) as ds:
            # *** 修正 ***: 检查 'STID' 维度的长度
            # (基于 summary.txt, 维度是 'STID')
            num_original = len(ds['STID'])
            print(f"  [i] 原始文件包含 {num_original} 个站点。")

            # --- 核心操作: *** 修正 *** ---
            # 使用 .sel() 按 'STID' 维度筛选
            print(f"  正在筛选 {len(good_station_ids)} 个站点...")
            ds_filtered = ds.sel(STID=good_station_ids)
            
            # *** 修正 ***: 检查筛选后的 'STID' 维度的长度
            num_filtered = len(ds_filtered['STID'])
            print(f"  [i] 筛选后文件包含 {num_filtered} 个站点。")

            # --- 3. 保存筛选后的 NC 文件 ---
            print(f"  正在保存到新文件: {output_nc_path}")
            
            # 保存到 NetCDF
            ds_filtered.to_netcdf(output_nc_path)
            
            num_removed = num_original - num_filtered
            print(f"\n✅ 成功! 已移除 {num_removed} 个低质量站点。")
            print(f"   (原始: {num_original}, 保留: {num_filtered})")
            print(f"   新文件已保存到:\n   {output_nc_path}")

    except FileNotFoundError:
        print(f"  [!!] 错误: 找不到原始 NetCDF 文件: {original_nc_path}")
    except KeyError as e:
        # *** 修正 ***: 捕捉 'STID' 错误
        if 'STID' in str(e):
            print(f"  [!!] 错误: 在 NetCDF 文件中找不到名为 'STID' 的维度。")
        elif 'not found in axis' in str(e):
            print(f"  [!!] 错误: CSV 中的 STID 与 NC 文件 'STID' 维度的坐标不匹配。")
            print(f"       {e}")
        else:
            print(f"  [!!] 筛选时发生KeyError: {e}")
    except Exception as e:
        print(f"  [!!] 发生未知错误: {e}")


# --- 脚本主程序入口 ---
if __name__ == "__main__":
    
    # --- 1. 文件路径设置 (与你的 '查看_站点质量.py' 保持一致) ---
    base_dir = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据_v2/"
    
    # 输入文件
    # (原始的、包含所有站点的 NC 文件)
    nc_file = os.path.join(base_dir, "Combine_Stations_ExMaxWind+SLP+StP_Fixed_2004_2024.nc")
    
    # (用于决策的质量报告)
    csv_quality_report = os.path.join("/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据_v2/4_合并后_区分缺测类型_按站点.csv")
    
    # 输出文件
    # (你想要的、精简后的新 NC 文件名)
    output_nc_file = os.path.join(base_dir, "Refined_Combine_Stations_ExMaxWind+SLP+StP_Fixed_2004_2024.nc")
    
    # --- 2. 运行筛选函数 ---
    # 阈值: 你希望保留缺测率 <= 10.0% 的站点
    filter_nc_by_quality(nc_file, csv_quality_report, output_nc_file, keep_threshold=10.0)