import xarray as xr
import numpy as np
import os

# --- 1. 设置你的文件路径 ---
# (请确保路径与你提供的完全一致)
old_file_path = "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/All_Typhoons_ExMaxWind.nc"
new_file_path = "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/Refined_Combine_Stations_ExMaxWind.nc"

def compare_station_lists(old_file, new_file):
    print("--- 站点对比开始 ---")
    
    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(old_file):
        print(f"错误: 找不到旧文件: {old_file}")
        return
    if not os.path.exists(new_file):
        print(f"错误: 找不到新文件: {new_file}")
        return
        
    print(f"正在读取旧文件 (国家站): {old_file}")
    print(f"正在读取新文件 (乡镇站): {new_file}\n")

    try:
        # --- 3. 使用 xarray 打开文件 ---
        # 'with' 语句确保文件在使用后被正确关闭
        with xr.open_dataset(old_file) as ds_old, xr.open_dataset(new_file) as ds_new:
            
            # --- 4. 提取 STID 坐标 ---
            # 假设两个文件中的站点坐标都命名为 'STID'
            # .values 会将其转换为 numpy 数组
            try:
                old_stations = ds_old['STID'].values
            except KeyError:
                print(f"错误: 旧文件中未找到名为 'STID' 的坐标。")
                print(f"  旧文件中的坐标有: {list(ds_old.coords)}")
                return
                
            try:
                new_stations = ds_new['STID'].values
            except KeyError:
                print(f"错误: 新文件中未找到名为 'STID' 的坐标。")
                print(f"  新文件中的坐标有: {list(ds_new.coords)}")
                return

            # --- 5. 转换为 Set (集合) 以便对比 ---
            # 转换为 str 类型确保对比的健壮性 (避免 bytes vs str 的问题)
            set_old = set(np.asarray(old_stations).astype(str))
            set_new = set(np.asarray(new_stations).astype(str))
            
            print(f"旧文件 (国家站) 站点数: {len(set_old)}")
            print(f"新文件 (乡镇站) 站点数: {len(set_new)}")
            
            # --- 6. 执行对比 ---
            # 检查旧集合是否是新集合的子集
            is_subset = set_old.issubset(set_new)
            
            if is_subset:
                print("\n✅ 检查结果: 是")
                print("所有 102 个国家站站点 均已包含 在新的 1146 个站点文件中。")
            else:
                print("\n❌ 检查结果: 否")
                print("并非所有国家站站点都包含在新文件中。")
                
                # 找出差异
                missing_stations = set_old.difference(set_new)
                common_stations = set_old.intersection(set_new)
                
                print(f"\n共同站点数: {len(common_stations)}")
                print(f"未在新文件中找到的国家站 (共 {len(missing_stations)} 个):")
                
                # 打印缺失的站点ID
                for station in sorted(list(missing_stations)):
                    print(f"  - {station}")

    except Exception as e:
        print(f"\n处理文件时发生意外错误: {e}")

# --- 运行脚本 ---
if __name__ == "__main__":
    compare_station_lists(old_file_path, new_file_path)
    print("\n--- 对比完成 ---")