import xarray as xr
import numpy as np
import os
import sys

def combine_authoritative_nc_files(file_path_national, file_path_township, output_filepath):
    """
    合并两个台风 NetCDF 文件。

    策略:
    - 信任 '国家站' (national) 文件的元数据。
    - 合并所有独有站点，并使用 '国家站' 文件的元数据和数据来覆盖 '重叠' 站点。
    """
    print("正在加载 NetCDF 文件...")
    print(f"  [权威源] 国家站: {file_path_national}")
    print(f"  [补充源] 乡镇站: {file_path_township}")

    try:
        with xr.open_dataset(file_path_national) as ds_nat, \
             xr.open_dataset(file_path_township) as ds_town:
            
            print("文件加载完毕。\n")

            # --- 1. 站点 (STID) 集合分析 ---
            print("=" * 80)
            print("1. 站点 (STID) 集合分析")
            print("=" * 80)
            
            stids_nat = set(np.char.strip(ds_nat['STID'].values.astype(str)))
            stids_town = set(np.char.strip(ds_town['STID'].values.astype(str)))

            common_stations = stids_nat.intersection(stids_town)
            national_only = stids_nat.difference(stids_town)
            township_only = stids_town.difference(stids_nat)
            
            total_final_stations = len(common_stations) + len(national_only) + len(township_only)

            print(f"  重叠站点 (使用国家站数据): {len(common_stations)} 个")
            print(f"  国家站独有站点: {len(national_only)} 个")
            print(f"  乡镇站独有站点: {len(township_only)} 个")
            print(f"  ---------------------------------")
            print(f"  最终合并站点总数: {total_final_stations} 个 ")
            
            if total_final_stations == 0:
                print("没有站点可合并。")
                return

            # --- 2. 准备三个数据组件 ---
            print("\n" + "=" * 80)
            print("2. 准备数据组件...")
            print("=" * 80)

            # 组件 1 & 2: 来自国家站 (权威源)
            # 包含了 64 个重叠站 + 38 个独有站
            stids_from_national = list(common_stations.union(national_only))
            ds_from_nat = ds_nat.sel(STID=stids_from_national)
            print(f"  [组件1+2] 从 '国家站' 文件提取 {len(stids_from_national)} 个站点 (重叠站+独有站)")

            # 组件 3: 来自乡镇站 (补充源)
            # 只包含 1082 个乡镇站独有的站
            stids_from_township = list(township_only)
            ds_from_town = ds_town.sel(STID=stids_from_township)
            print(f"  [组件3]   从 '乡镇站' 文件提取 {len(stids_from_township)} 个站点 (独有站)")

            # --- 3. 合并和保存 ---
            print("\n" + "=" * 80)
            print("3. 合并并保存文件...")
            print("=" * 80)

            # 沿着 'STID' 维度合并
            # ds_from_nat 放在第一个，以确保保留其全局属性 (如台风映射)
            ds_combined = xr.concat([ds_from_nat, ds_from_town], dim="STID")
            
            print(f"  成功合并。 维度: {ds_combined.dims}")

            # 按 STID 排序，使文件更整洁
            ds_final = ds_combined.sortby("STID")
            print(f"  已按 STID 排序。")

            # 检查最终变量形状
            final_shape = ds_final['wind_velocity'].shape
            print(f"  最终 'wind_velocity' 变量形状: {final_shape}")
            if final_shape == (3944, 1, total_final_stations):
                 print(f"  [✓] 形状正确 (时间, TL, 站点)")
            else:
                 print(f"  [!] 警告: 最终形状 {final_shape} ")


            # 定义输出文件的编码 (与源文件保持一致)
            encoding = {
                "wind_velocity": {"dtype": "float32", "zlib": True, "complevel": 5, "_FillValue": np.nan},
                "wind_direction": {"dtype": "int16", "zlib": True, "complevel": 5, "_FillValue": -999},
                "typhoon_id_index": {"dtype": "int32", "zlib": True, "complevel": 5, "_FillValue": -999},
                # 坐标和其他变量 xarray 会自动处理
            }
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_filepath)
            if output_dir: # 检查是否为空字符串
                os.makedirs(output_dir, exist_ok=True)
                
            # 保存到 NetCDF
            ds_final.to_netcdf(output_filepath, encoding=encoding)
            
            print(f"\n✅ 成功! 合并后的nc文件已保存到:")
            print(f"   {output_filepath}")

    except FileNotFoundError as e:
        print(f"[错误] 文件未找到: {e.filename}")
    except Exception as e:
        import traceback
        print(f"合并过程中发生意外错误: {e}")
        traceback.print_exc()

# --- 脚本主程序入口 ---
if __name__ == "__main__":
    # *** 请根据你的实际路径修改这里 ***
    
    # 输入文件
    path_national = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据/All_Typhoons_ExMaxWind_2004_2024.nc"
    path_township = '/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据/Representative_Stations_Typhoons_ExMaxWind_Fixed_2004_2024.nc'

    # 输出文件
    path_output = "/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据/Combine_Stations_ExMaxWind_Fixed_2004_2024.nc"
    
    combine_authoritative_nc_files(path_national, path_township, path_output)