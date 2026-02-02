"""
ERA5_500hPa_Download_V11_FixConnection.py

修复说明：
1. [关键修复] 自动清洗 CDS_URL 和 CDS_KEY，去除可能误填的 "url: " 或 "key: " 前缀，解决 Connection Adapter 报错。
2. [数据集确认] 500hPa 高度场必须使用 'reanalysis-era5-pressure-levels'。
   - 如果您需要的是地面数据（如10m风），请在代码中修改 DATASET 变量，但这里默认按您之前的需求（500hPa）配置。

输入：AllTyphoons_Max_Summary.xlsx
输出：ERA5_Data_Precise/Output_Typhoons_NC/
"""

import cdsapi
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# ======= 配置区域 (请在此处填入您的 Key) =======

# 1. 输入 Excel 路径
INPUT_EXCEL = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分级统计/输出_台风过程分时段统计/AllTyphoons_Max_Summary.xlsx"

# 2. 根输出目录
BASE_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/ERA5_Data_10级及以上")

# 3. CDS API 配置 
CDS_URL = "https://cds.climate.copernicus.eu/api" 
CDS_KEY = "6a83cd0d-c34c-4fd1-ab54-24e2a59a163b" 

# 4. 下载参数
# 500hPa 高度场必须用 pressure-levels
DATASET = 'reanalysis-era5-pressure-levels' 
VARIABLE = 'geopotential'
PRESSURE_LEVEL = '500'

# 如果您想改下地面数据(single-levels)，请解注下面几行，并注释上面三行：
# DATASET = 'reanalysis-era5-single-levels'
# VARIABLE = 'mean_sea_level_pressure' # 或 '10m_u_component_of_wind' 等
# PRESSURE_LEVEL = None 

# 下载区域 [North, West, South, East]
AREA = [40, 110, 20, 135] 

# ========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_cds_config(url, key):
    """清理配置字符串，防止 'url: https...' 导致的连接错误"""
    if url:
        url = url.replace("url:", "").strip()
    if key:
        key = key.replace("key:", "").strip()
    return url, key

def get_typhoon_time_mapping(excel_path, sheets):
    print(f"正在读取 Excel: {excel_path}")
    mapping_list = []
    unique_times = set()
    
    try:
        xls = pd.ExcelFile(excel_path)
        for sheet in sheets:
            if sheet not in xls.sheet_names:
                continue
            
            print(f"  -> 读取 Sheet: {sheet}")
            df = pd.read_excel(xls, sheet_name=sheet)
            
            # 兼容列名可能的空格
            df.columns = [c.strip() for c in df.columns]
            
            required_cols = ['中文名', '英文名', '出现时刻']
            if not all(col in df.columns for col in required_cols):
                print(f"    [Skip] {sheet} 缺少必要列")
                continue
            
            for idx, row in df.iterrows():
                try:
                    t_str = str(row['出现时刻'])
                    # 转换时间 CST -> UTC
                    dt_cst = pd.to_datetime(t_str)
                    dt_utc = dt_cst - timedelta(hours=8)
                    
                    unique_times.add(dt_utc)
                    
                    mapping_list.append({
                        "sheet": sheet,
                        "cn_name": str(row['中文名']).strip(),
                        "en_name": str(row['英文名']).strip(),
                        "utc_time": dt_utc
                    })
                except:
                    pass
    except Exception as e:
        print(f"[Error] 读取 Excel 失败: {e}")
        return [], []

    sorted_times = sorted(list(unique_times))
    print(f"[Info] 解析完毕。需下载 {len(sorted_times)} 个唯一 UTC 时刻。")
    return mapping_list, sorted_times

def download_precise_times(time_list, cache_dir, url, key):
    # 初始化 Client
    if url and key:
        c = cdsapi.Client(url=url, key=key)
    else:
        # 如果未提供，尝试读取 .cdsapirc 文件
        try:
            c = cdsapi.Client()
        except Exception as e:
            print(f"[Fatal Error] 无法初始化 CDS Client，请检查 Key 配置: {e}")
            return

    ensure_dir(cache_dir)
    total = len(time_list)
    print(f"\n>>> 开始精确下载 ({total} 个文件)...")
    print(f"    数据集: {DATASET}")
    print(f"    变量: {VARIABLE} (Level: {PRESSURE_LEVEL})")
    
    for i, dt in enumerate(time_list):
        fname = f"ERA5_{dt.strftime('%Y%m%d_%H')}.nc"
        fpath = cache_dir / fname
        progress = f"[{i+1}/{total}]"
        
        if fpath.exists():
            print(f"  {progress} {fname} 已存在，跳过。")
            continue
            
        print(f"  {progress} 正在下载: {dt} (UTC) ...")
        
        # 构建请求字典
        request_params = {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': VARIABLE,
            'year': dt.strftime("%Y"),
            'month': dt.strftime("%m"),
            'day': dt.strftime("%d"),
            'time': dt.strftime("%H:00"),
            'area': AREA,
        }
        
        # 只有 pressure-levels 数据集才加 pressure_level 参数
        if 'pressure-levels' in DATASET and PRESSURE_LEVEL:
            request_params['pressure_level'] = PRESSURE_LEVEL

        try:
            c.retrieve(DATASET, request_params, str(fpath))
        except Exception as e:
            print(f"    [Error] 下载失败: {e}")

def distribute_files(mapping_list, cache_dir, output_dir):
    print(f"\n>>> 正在生成台风独立文件...")
    ensure_dir(output_dir)
    
    for item in mapping_list:
        dt = item['utc_time']
        cache_name = f"ERA5_{dt.strftime('%Y%m%d_%H')}.nc"
        cache_path = cache_dir / cache_name
        
        if not cache_path.exists():
            continue
            
        safe_cn = item['cn_name'].replace('/', '_')
        safe_en = item['en_name'].replace('/', '_')
        safe_sheet = item['sheet'].replace('/', '_')
        time_str = dt.strftime('%Y%m%d_%H')
        
        target_name = f"{safe_cn}_{safe_en}_{safe_sheet}_{time_str}_UTC.nc"
        target_path = output_dir / target_name
        
        try:
            shutil.copy2(cache_path, target_path)
        except Exception as e:
            print(f"  [Error] 复制失败: {e}")
            
    print(f"[Success] 处理完成！文件位于: {output_dir}")

def main():
    # 清洗配置
    clean_url, clean_key = clean_cds_config(CDS_URL, CDS_KEY)
    
    # 路径
    CACHE_DIR = BASE_DIR / "Cache_Time_Files"
    FINAL_DIR = BASE_DIR / "Output_Typhoons_NC"
    
    mapping_list, unique_times = get_typhoon_time_mapping(INPUT_EXCEL, ["10级及以上"])
    
    if unique_times:
        download_precise_times(unique_times, CACHE_DIR, clean_url, clean_key)
        distribute_files(mapping_list, CACHE_DIR, FINAL_DIR)

if __name__ == "__main__":
    main()