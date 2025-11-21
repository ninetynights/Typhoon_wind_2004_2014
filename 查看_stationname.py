import xarray as xr
import numpy as np

def check_station_name_encoding(file_path):
    """
    检查 NetCDF 文件中 'station_name' 变量是否存在编码问题 (乱码)。
    """
    print(f"正在打开文件进行编码检查: {file_path}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            if 'station_name' not in ds:
                print("[错误] 文件中未找到 'station_name' 变量。")
                return
                
            station_names = ds['station_name'].values
            total_count = len(station_names)
            garbled_count = 0
            
            print(f"开始检查所有 {total_count} 个站点名称...")
            
            # 定义一个函数来检测是否为乱码
            # 我们的乱码特征是包含了 U+00A0 到 U+00FF 范围内的字符
            def is_garbled(name_str):
                for char in name_str:
                    # 这个范围 (Latin-1 Supplement) 是GBK误读为Latin-1的典型特征
                    if '\u00A0' <= char <= '\u00FF': 
                        return True
                return False

            garbled_samples = []
            
            for name in station_names:
                # np.char.strip 用于去除可能的空白符
                name_str = str(np.char.strip(name)) 
                if is_garbled(name_str):
                    garbled_count += 1
                    if len(garbled_samples) < 5: # 只保存前5个样本
                        garbled_samples.append(name_str)
                        
            print("\n" + "=" * 80)
            print("编码检查结果")
            print("=" * 80)
            print(f"文件总站点数: {total_count}")
            print(f"检测到乱码的站点数: {garbled_count}")
            
            if garbled_count == total_count:
                print("\n  [!!] 结论: 是的，该文件中所有的 (100%) 站点名称似乎都是乱码。")
            elif garbled_count > 0:
                print(f"\n  [!] 结论: 该文件中有 {garbled_count} 个站点名称被识别为乱码。")
            else:
                print("\n  [✓] 结论: 未检测到明显乱码。")
                
            if garbled_samples:
                print("\n前5个乱码样本:")
                for sample in garbled_samples:
                    print(f"  - '{sample}'")

    except Exception as e:
        print(f"分析过程中发生意外错误: {e}")

# --- 脚本主程序入口 ---
if __name__ == "__main__":
    # *** 只需要乡镇站文件的路径 ***
    path_township = "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/Refined_Combine_Stations_ExMaxWind_Fixed.nc"
    
    check_station_name_encoding(path_township)