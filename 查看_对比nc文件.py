import xarray as xr
import numpy as np
import pandas as pd
import os
import sys

def compare_typhoon_nc_files(file_path_national, file_path_township, output_dir):
    """
    对比分析两个台风 NetCDF 文件 (国家站 vs. 乡镇站)。
    将详细的对比报告保存到指定的 output_dir 文件夹中的 CSV 文件。
    """
    print("正在加载 NetCDF 文件...")
    print(f"  国家站文件: {file_path_national}")
    print(f"  乡镇站文件: {file_path_township}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"对比报告将保存到: {output_dir}")

    # 使用 'with' 语句同时打开两个文件
    try:
        with xr.open_dataset(file_path_national) as ds_nat, \
             xr.open_dataset(file_path_township) as ds_town:
            
            print("文件加载完毕。\n")

            # --- 1. 站点 (STID) 集合分析 ---
            print("=" * 80)
            print("1. 站点 (STID) 集合分析")
            print("=" * 80)
            
            try:
                stids_nat = set(np.char.strip(ds_nat['STID'].values.astype(str)))
                stids_town = set(np.char.strip(ds_town['STID'].values.astype(str)))
            except KeyError:
                print("[错误] 'STID' 变量在文件中未找到。")
                return

            common_stations = stids_nat.intersection(stids_town)
            national_only = stids_nat.difference(stids_town)
            township_only = stids_town.difference(stids_nat)

            common_stations_list = sorted(list(common_stations))
            national_only_list = sorted(list(national_only))
            township_only_list = sorted(list(township_only))
            
            print(f"国家站 (All_Typhoons) 文件中站点数量: {len(stids_nat)}")
            print(f"乡镇站 (New_Stations) 文件中站点数量: {len(stids_town)}")
            print(f"\n共同(重叠)的站点数量: {len(common_stations)}")
            print(f"仅在 '国家站' 文件中存在的站点数量: {len(national_only)}")
            print(f"仅在 '乡镇站' 文件中存在的站点数量: {len(township_only)}")

            # --- 保存 CSV 1: 站点集合分析 ---
            s_common = pd.Series(common_stations_list, name="common_stations")
            s_nat_only = pd.Series(national_only_list, name="national_only_stations")
            s_town_only = pd.Series(township_only_list, name="township_only_stations")
            
            df_overlap = pd.concat([s_common, s_nat_only, s_town_only], axis=1)
            csv_path1 = os.path.join(output_dir, "1_站点集合分析.csv")
            df_overlap.to_csv(csv_path1, index=False, encoding='utf-8-sig')
            print(f"\n  [✓] 站点集合列表已保存到: {csv_path1}")


            if not common_stations:
                print("\n两个文件没有共同的站点ID。对比结束。")
                return

            # --- 2. 重叠站点元数据 (Lat, Lon, Height) 对比 ---
            print("\n" + "=" * 80)
            print("2. 重叠站点元数据 (Lat, Lon, Height) 对比")
            print("=" * 80)
            
            metadata_mismatches = []
            
            # 遍历所有重叠站点
            for stid in common_stations_list:
                meta_nat = ds_nat.sel(STID=stid)
                meta_town = ds_town.sel(STID=stid)
                
                lat_nat, lon_nat, hgt_nat = meta_nat['lat'].item(), meta_nat['lon'].item(), meta_nat['height'].item()
                lat_town, lon_town, hgt_town = meta_town['lat'].item(), meta_town['lon'].item(), meta_town['height'].item()

                if not np.isclose(lat_nat, lat_town) or \
                   not np.isclose(lon_nat, lon_town) or \
                   not np.isclose(hgt_nat, hgt_town):
                    
                    # 记录所有不匹配的细节
                    metadata_mismatches.append({
                        "STID": stid,
                        "lat_national": lat_nat,
                        "lat_township": lat_town,
                        "lon_national": lon_nat,
                        "lon_township": lon_town,
                        "height_national": hgt_nat,
                        "height_township": hgt_town
                    })
            
            if not metadata_mismatches:
                print("  [✓] 所有 (64 个) 重叠站点的元数据 (Lat, Lon, Height) 完全一致。")
            else:
                print(f"\n  [!] 总结: 总共有 {len(metadata_mismatches)} 个重叠站点的元数据不匹配。")
                # --- 保存 CSV 2: 元数据差异对比 ---
                df_meta_mismatch = pd.DataFrame(metadata_mismatches)
                csv_path2 = os.path.join(output_dir, "2_元数据差异对比.csv")
                df_meta_mismatch.to_csv(csv_path2, index=False, encoding='utf-8-sig')
                print(f"  [✓] 元数据不匹配详情已保存到: {csv_path2}")


            # --- 3. 重叠站点数据值 (wind_velocity) 全量对比 ---
            print("\n" + "=" * 80)
            print("3. 重叠站点数据值 (wind_velocity) 全量对比")
            print("=" * 80)
            
            data_comparison_results = []
            
            # 检查时间轴是否一致 (只需要检查一次)
            if not np.array_equal(ds_nat['INITTIME'].values, ds_town['INITTIME'].values):
                print("[!!] 严重错误: 两个文件的时间轴 (INITTIME) 不一致，无法进行数据对比。")
                return

            print(f"正在遍历所有 {len(common_stations_list)} 个重叠站点，对比 'wind_velocity' 时间序列...")
            
            for stid in common_stations_list:
                ts_nat = ds_nat['wind_velocity'].sel(STID=stid).squeeze()
                ts_town = ds_town['wind_velocity'].sel(STID=stid).squeeze()
                
                diff = np.abs(ts_nat - ts_town)
                valid_diff = diff.where(np.isfinite(ts_nat) & np.isfinite(ts_town))
                
                max_diff = valid_diff.max().item()
                mean_diff = valid_diff.mean().item()
                
                if np.isnan(max_diff):
                    max_diff = 0.0 # 如果都是 NaN, 差异视为 0
                    mean_diff = 0.0
                
                # 统计差异不为0的时间点数量
                non_zero_diff_count = int(np.count_nonzero(valid_diff.fillna(0)))
                
                data_comparison_results.append({
                    "STID": stid,
                    "is_identical": bool(non_zero_diff_count == 0),
                    "different_timesteps_count": non_zero_diff_count,
                    "max_abs_difference_ms": max_diff,
                    "mean_abs_difference_ms": mean_diff
                })

            # --- 保存 CSV 3: 数据值差异对比 ---
            df_data_comp = pd.DataFrame(data_comparison_results)
            csv_path3 = os.path.join(output_dir, "3_数据值差异对比.csv")
            df_data_comp.to_csv(csv_path3, index=False, encoding='utf-8-sig')
            
            total_mismatched_data_stations = df_data_comp['is_identical'].count() - df_data_comp['is_identical'].sum()
            
            if total_mismatched_data_stations == 0:
                print("\n  [✓] 巨大好消息! 所有 64 个重叠站点的 'wind_velocity' 数据完全一致。")
            else:
                print(f"\n  [!!] 严重警告: {len(common_stations_list)} 个重叠站中，有 {total_mismatched_data_stations} 个站点的 'wind_velocity' 数据存在差异。")
            
            print(f"  [✓] 'wind_velocity' 全量对比报告已保存到: {csv_path3}")

            # --- 4. 站点名称 (station_name) 抽样检查 (保持) ---
            print("\n" + "=" * 80)
            print("4. 站点名称 (station_name) 编码抽样检查")
            print("=" * 80)

            check_stid = '58443'
            if check_stid in common_stations:
                name_nat = ds_nat['station_name'].sel(STID=check_stid).item()
                name_town = ds_town['station_name'].sel(STID=check_stid).item()
                print(f"抽样对比 (站点 {check_stid}):")
                print(f"  '国家站' 文件 (All_Typhoons) 名称: '{name_nat}'")
                print(f"  '乡镇站' 文件 (New_Stations) 名称: '{name_town}'")
                if name_nat == name_town:
                    print("\n  [✓] 该站点的名称一致。 (编码问题已解决)")
                else:
                    print(f"\n  [!] 发现: 两个文件的站点名称不一致 ('{name_nat}' vs '{name_town}')。")

    except FileNotFoundError as e:
        print(f"[错误] 文件未找到: {e.filename}")
    except Exception as e:
        import traceback
        print(f"分析过程中发生意外错误: {e}")
        traceback.print_exc()

# --- 脚本主程序入口 ---
if __name__ == "__main__":
    # *** 请根据你的实际路径修改这里 ***
    # 国家站站点
    path_national = "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/All_Typhoons_ExMaxWind.nc"
    # 乡镇站点
    path_township = "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/Representative_Stations_Typhoons_ExMaxWind.nc"
    
    # *** 新增：指定一个输出目录来存放 CSV 报告 ***
    output_report_dir = "/Users/momo/Desktop/业务相关/2025 影响台风大风/对比报告"
    
    compare_typhoon_nc_files(path_national, path_township, output_report_dir)