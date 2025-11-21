
"""
总体说明：
- 目的：快速检查并汇总观测型 NetCDF（按站点×时间×层/要素）中的缺测情况与基本元信息，
  并生成人可读的文本摘要与两份 CSV 报告（按站点、按时间）。
- 主要功能：
  1) 打印文件维度、站点数、时间点数等基本信息（输出到 info_summary.txt）；
  2) 针对变量 'wind_velocity' 做缺测诊断：总体缺测率、按站点缺测率、按时间缺测率；
  3) 将按站点与按时间的完整缺测表分别保存为 CSV，且在 TXT 中给出 Top10 摘要；
  4) 对可能的异常或缺失变量有容错处理并记录警告信息。
- 输入/输出：
  - 输入：在脚本顶部的 file_path 变量指定要检查的 NetCDF 文件路径。
  - 输出（与脚本同目录）：
      * {basename}_info_summary.txt    （TXT，包含打印的诊断信息）
      * 4_合并后缺测报告_按站点.csv      （按站点完整缺测率表）
      * 5_合并后缺测报告_按时间.csv      （按时间完整缺测率表）

"""

import xarray as xr
import pandas as pd
import os
import sys

# --- 1. 定义文件路径 ---
file_path = '/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据/Representative_Stations_Typhoons_ExMaxWind_Fixed_2004_2024.nc'

# --- 2. 自动生成输出文件路径 ---
base_dir = os.path.dirname(file_path)
base_name = os.path.splitext(os.path.basename(file_path))[0]

output_txt_path = os.path.join(base_dir, f"{base_name}_info_summary.txt")
output_csv_station = os.path.join(base_dir, "4_代表站_缺测报告_按站点.csv")
output_csv_time = os.path.join(base_dir, "5_代表站_缺测报告_按时间.csv")


# --- 3. 重定向标准输出 (stdout) ---
print(f"正在分析文件: {file_path}")
print(f"分析摘要将保存到: {output_txt_path}")
print(f"按站点缺测报告将保存到: {output_csv_station}")
print(f"按时间缺测报告将保存到: {output_csv_time}")

original_stdout = sys.stdout

try:
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        sys.stdout = f  # 将标准输出重定向到 TXT 文件

        # 打开NetCDF文件
        with xr.open_dataset(file_path) as ds:
            
            # === 1. 文件基本信息 (打印到 TXT) ===
            print("=" * 80)
            print(f"NetCDF文件基本信息: {file_path}")
            print("=" * 80)
            print(f"文件维度: {list(ds.dims)}")
            if 'STID' in ds:
                stations = ds['STID'].values
                print(f"\n站点数量: {len(stations)}")
            if 'INITTIME' in ds:
                times = ds['INITTIME'].values
                print(f"\n时间点数量: {len(times)}")

            # === 2. 缺测数据分析 (打印摘要到 TXT + 保存完整 CSV) ===
            print("\n" + "=" * 80)
            print("缺测数据分析 (Missing Data Analysis for 'wind_velocity')")
            print("=" * 80)
            
            if 'wind_velocity' in ds:
                print("  正在加载 'wind_velocity' 数据进行分析... (这可能需要一些时间)")
                wind_data = ds['wind_velocity'].load()
                print("  数据加载完毕。")
                
                # 1. 总体缺测 (打印到 TXT)
                total_points = wind_data.size
                missing_points = int(wind_data.isnull().sum())
                missing_percent = (missing_points / total_points) * 100
                print("\n  --- 1. 总体缺测情况 ---")
                print(f"  总体缺测百分比: {missing_percent:.2f}%")
                
                # 2. 按站点分析
                print("\n  --- 2. 按站点缺测分析 ---")
                missing_by_station = wind_data.isnull().sum(dim=['INITTIME', 'TL'])
                total_time_points = wind_data.shape[0] * wind_data.shape[1]
                percent_missing_by_station = (missing_by_station / total_time_points) * 100
                
                # <--- 修正点 1 (修复 TypeError) ---
                # 先转换为 Series，然后再设置 name 属性
                percent_missing_by_station_pd = percent_missing_by_station.to_series()
                percent_missing_by_station_pd.name = "missing_percent"
                # <--- 修正结束 ---
                percent_missing_by_station_pd.index.name = "STID"
                
                # (打印 Top 10 摘要到 TXT)
                print("    [!] 缺测最多的10个站点 (数据质量最差):")
                for stid, percent in percent_missing_by_station_pd.nlargest(10).items():
                    print(f"      - 站点 {stid}: 缺测 {percent:.2f}%")
                print("\n    [✓] 缺测最少的10个站点 (数据质量最好):")
                for stid, percent in percent_missing_by_station_pd.nsmallest(10).items():
                    print(f"      - 站点 {stid}: 缺测 {percent:.2f}%")
                
                # 保存完整的站点缺测报告到 CSV
                try:
                    percent_missing_by_station_pd.sort_values(ascending=False).to_csv(output_csv_station, encoding='utf-8-sig')
                    print(f"\n    [✓] 成功: 缺测报告已保存到 {output_csv_station}")
                except Exception as e_csv:
                    print(f"\n    [!] 错误: 无法保存站点缺测 CSV 文件: {e_csv}")
                
                
                # 3. 按时间分析
                print("\n  --- 3. 按时间缺测分析 ---")
                missing_by_time = wind_data.isnull().sum(dim=['STID', 'TL'])
                total_stations = wind_data.shape[1] * wind_data.shape[2]
                percent_missing_by_time = (missing_by_time / total_stations) * 100
                
                # <--- 修正点 2 (修复 TypeError) ---
                # 同样，先转换，后命名
                percent_missing_by_time_pd = percent_missing_by_time.to_series()
                percent_missing_by_time_pd.name = "missing_percent"
                # <--- 修正结束 ---
                percent_missing_by_time_pd.index.name = "INITTIME"

                # (打印 Top 10 摘要到 TXT)
                print("\n    [!] 缺测最严重的10个时间点 (站点覆盖率最低):")
                for time_val, percent in percent_missing_by_time_pd.nlargest(10).items():
                    time_str = pd.to_datetime(time_val).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"      - 时间 {time_str}: {percent:.2f}% 的站点缺测")

                # 保存完整的时间缺测报告到 CSV
                try:
                    percent_missing_by_time_pd.sort_values(ascending=False).to_csv(output_csv_time, encoding='utf-8-sig')
                    print(f"\n    [✓] 成功: 缺测报告已保存到 {output_csv_time}")
                except Exception as e_csv:
                    print(f"\n    [!] 错误: 无法保存时间缺测 CSV 文件: {e_csv}")
            
            else:
                print("  [!] 错误: 未在文件中找到 'wind_velocity' 变量。")

            # ... (你原始脚本的其他部分，如台风映射等) ...

finally:
    # --- 4. 恢复标准输出 ---
    sys.stdout = original_stdout

print("--- 分析完成! ---")