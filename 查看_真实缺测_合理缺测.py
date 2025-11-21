# -*- coding: utf-8 -*-
"""
总体说明：
- 目的：检查 NetCDF 文件，并区分统计 "建站前缺失(NaN)" 和 "真实缺测(-999)"。
- 核心逻辑：
  1. Pre-observation (未建站): 值为 NaN
  2. True Missing (真实缺测): 值为 -999
  3. Valid Data (有效数据): 其他值
- 输出：
  - {basename}_info_summary.txt: 文本摘要
  - 4_区分缺测类型_按站点.csv: 包含未建站率、真实缺测率、业务缺测率
  - 5_区分缺测类型_按时间.csv: 按时间点的统计
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import sys

# --- 1. 定义文件路径 ---
file_path = r'/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据/Combine_Stations_ExMaxWind_Fixed_2004_2024.nc'

# --- 2. 自动生成输出文件路径 ---
base_dir = os.path.dirname(file_path)
base_name = os.path.splitext(os.path.basename(file_path))[0]

output_txt_path = os.path.join(base_dir, f"{base_name}_详细质量分析报告.txt")
output_csv_station = os.path.join(base_dir, "4_合并后_区分缺测类型_按站点.csv")
output_csv_time = os.path.join(base_dir, "5_合并后_区分缺测类型_按时间.csv")

# --- 3. 辅助函数：重定向打印 ---
print(f"正在分析文件: {file_path}")
print(f"分析摘要将保存到: {output_txt_path}")

original_stdout = sys.stdout

try:
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        sys.stdout = f  # 将标准输出重定向到 TXT 文件

        # 打开NetCDF文件
        with xr.open_dataset(file_path) as ds:
            
            # === 1. 文件基本信息 ===
            print("=" * 80)
            print(f"NetCDF文件基本信息: {base_name}")
            print("=" * 80)
            
            stations = ds['STID'].values if 'STID' in ds else []
            times = ds['INITTIME'].values if 'INITTIME' in ds else []
            
            print(f"文件维度: {list(ds.dims)}")
            print(f"站点数量: {len(stations)}")
            print(f"时间点数: {len(times)}")
            print(f"时间范围: {pd.to_datetime(times[0])} 至 {pd.to_datetime(times[-1])}")

            # === 2. 详细缺测数据分析 ===
            print("\n" + "=" * 80)
            print("详细数据质量分析 (Data Quality Analysis)")
            print("逻辑定义:")
            print("  1. 未建站/无记录 (Pre-obs): 值 = NaN (客观缺失)")
            print("  2. 真实缺测 (True Missing): 值 = -999 (设备/传输故障)")
            print("  3. 有效数据 (Valid):        值 != NaN 且 != -999")
            print("=" * 80)
            
            if 'wind_velocity' in ds:
                print("  正在加载 'wind_velocity' 数据矩阵... ")
                wind_data = ds['wind_velocity'].load()
                
                # --- 生成掩码 (Masks) ---
                # Mask 1: 未建站 (NaN)
                mask_pre_obs = np.isnan(wind_data)
                
                # Mask 2: 真实缺测 (-999)
                # 注意: 浮点数比较建议用 isclose 或范围，但此处 -999 是显式赋值，直接比较通常可行
                # 为保险起见，处理 -999.0
                mask_true_missing = (wind_data == -999)
                
                # Mask 3: 有效数据
                mask_valid = (~mask_pre_obs) & (~mask_true_missing)
                
                # -------------------------------------------------
                # 1. 总体统计 (Global Stats)
                # -------------------------------------------------
                total_points = wind_data.size
                count_pre_obs = mask_pre_obs.sum().item()
                count_true_missing = mask_true_missing.sum().item()
                count_valid = mask_valid.sum().item()
                
                print("\n  --- 1. 总体数据构成 ---")
                print(f"  总数据点数: {total_points}")
                print(f"  [1] 未建站/无记录 (NaN): {count_pre_obs:<10} ({count_pre_obs/total_points*100:.2f}%)")
                print(f"  [2] 真实缺测 (-999):     {count_true_missing:<10} ({count_true_missing/total_points*100:.2f}%)")
                print(f"  [3] 有效数据 (Valid):    {count_valid:<10} ({count_valid/total_points*100:.2f}%)")
                
                # 计算"业务运行期间的缺测率" (排除未建站时期)
                operational_total = count_true_missing + count_valid
                if operational_total > 0:
                    ops_missing_rate = (count_true_missing / operational_total) * 100
                    print(f"\n  ** 全网平均业务缺测率 (仅统计建站后): {ops_missing_rate:.2f}% **")
                
                # -------------------------------------------------
                # 2. 按站点分析 (Station Stats)
                # -------------------------------------------------
                print("\n  --- 2. 按站点质量分析 ---")
                
                # 计算每个站点的各状态计数 (沿时间轴求和)
                # dims: (INITTIME, TL, STID) -> sum over INITTIME, TL
                st_n_pre = mask_pre_obs.sum(dim=['INITTIME', 'TL']).to_series()
                st_n_miss = mask_true_missing.sum(dim=['INITTIME', 'TL']).to_series()
                st_n_valid = mask_valid.sum(dim=['INITTIME', 'TL']).to_series()
                
                # 构建 DataFrame
                df_st = pd.DataFrame({
                    'Count_PreObs': st_n_pre,
                    'Count_TrueMissing': st_n_miss,
                    'Count_Valid': st_n_valid
                })
                
                df_st.index.name = 'STID'
                total_time_steps = wind_data.shape[0] * wind_data.shape[1]
                
                # 计算百分比 (占总时间轴)
                df_st['Pct_PreObs'] = (df_st['Count_PreObs'] / total_time_steps) * 100
                df_st['Pct_TrueMissing_Global'] = (df_st['Count_TrueMissing'] / total_time_steps) * 100
                df_st['Pct_Valid_Global'] = (df_st['Count_Valid'] / total_time_steps) * 100
                
                # *** 关键指标: 业务缺测率 ***
                # 公式: 缺测 / (缺测 + 有效)
                # 意义: 排除还没建站的时间，只看建站后坏了多久
                df_st['Operational_Total'] = df_st['Count_TrueMissing'] + df_st['Count_Valid']
                df_st['Operational_Missing_Rate'] = df_st.apply(
                    lambda row: (row['Count_TrueMissing'] / row['Operational_Total'] * 100) 
                    if row['Operational_Total'] > 0 else 0.0, axis=1
                )

                # 打印摘要
                print("    [!] 业务缺测率最高的10个站点 (建站后最不稳):")
                top10_bad = df_st.sort_values('Operational_Missing_Rate', ascending=False).head(10)
                for stid, row in top10_bad.iterrows():
                    print(f"      - 站点 {stid}: 业务缺测率 {row['Operational_Missing_Rate']:.2f}% "
                          f"(总时长中 {row['Pct_PreObs']:.1f}% 未建站)")

                print("\n    [i] 建站时间最短的10个站点 (NaN占比最高):")
                top10_new = df_st.sort_values('Pct_PreObs', ascending=False).head(10)
                for stid, row in top10_new.iterrows():
                    print(f"      - 站点 {stid}: 未建站占比 {row['Pct_PreObs']:.2f}%")

                # 保存
                try:
                    # 整理列顺序
                    out_cols = ['Pct_PreObs', 'Pct_Valid_Global', 'Pct_TrueMissing_Global', 'Operational_Missing_Rate', 
                                'Count_PreObs', 'Count_TrueMissing', 'Count_Valid']
                    df_st[out_cols].sort_values('Operational_Missing_Rate', ascending=False).to_csv(output_csv_station, encoding='utf-8-sig')
                    print(f"\n    [✓] 站点报告已保存: {output_csv_station}")
                except Exception as e:
                    print(f"    保存失败: {e}")

                # -------------------------------------------------
                # 3. 按时间分析 (Time Stats)
                # -------------------------------------------------
                print("\n  --- 3. 按时间质量分析 ---")
                
                # 沿站点轴求和
                tm_n_pre = mask_pre_obs.sum(dim=['STID', 'TL']).to_series()
                tm_n_miss = mask_true_missing.sum(dim=['STID', 'TL']).to_series()
                tm_n_valid = mask_valid.sum(dim=['STID', 'TL']).to_series()
                
                df_tm = pd.DataFrame({
                    'Count_PreObs': tm_n_pre,
                    'Count_TrueMissing': tm_n_miss,
                    'Count_Valid': tm_n_valid
                })
                
                df_tm.index.name = 'INITTIME'
                total_stations_count = len(stations)
                
                # 计算百分比
                df_tm['Pct_Stations_NotBuilt'] = (df_tm['Count_PreObs'] / total_stations_count) * 100
                df_tm['Pct_Stations_Missing'] = (df_tm['Count_TrueMissing'] / total_stations_count) * 100
                df_tm['Pct_Stations_Valid'] = (df_tm['Count_Valid'] / total_stations_count) * 100

                # 打印摘要
                print("    [!] 全网大面积缺测时刻 (Pct_Stations_Missing Top 10):")
                # 也就是当时已建站，但大面积坏了
                for idx, row in df_tm.nlargest(10, 'Pct_Stations_Missing').iterrows():
                    t_str = pd.to_datetime(idx).strftime('%Y-%m-%d %H:%M')
                    print(f"      - 时间 {t_str}: {row['Pct_Stations_Missing']:.2f}% 站点缺测 "
                          f"(有效率: {row['Pct_Stations_Valid']:.2f}%)")

                # 保存
                try:
                    df_tm.sort_index().to_csv(output_csv_time, encoding='utf-8-sig')
                    print(f"\n    [✓] 时间报告已保存: {output_csv_time}")
                except Exception as e:
                    print(f"    保存失败: {e}")
            
            else:
                print("  [!] 错误: 未找到 'wind_velocity' 变量")

finally:
    sys.stdout = original_stdout

print("--- 分析完成，请查看生成的 txt 和 csv 文件 ---")