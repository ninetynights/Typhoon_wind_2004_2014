"""
ERA5_Download_V14_SingleVar_Parallel.py

功能：
1. [轻量化] 仅下载 500hPa 位势高度 (Geopotential)，文件极小 (~35KB)。
2. [极速] 使用 5 线程并发下载，大幅缩短总排队时间。
3. [稳定] 修正了 API 参数，适配新版 CDS。

输入：AllTyphoons_Max_Summary.xlsx
输出：ERA5_Data_Single/Output_Typhoons_NC/
"""

import cdsapi
import pandas as pd
import shutil
import os
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======= 配置区域 =======
INPUT_EXCEL = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分级统计/输出_台风过程分时段统计/AllTyphoons_Max_Summary.xlsx"
BASE_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/ERA5_Data_Single_Fast") # 建议用新目录


CDS_URL = "https://cds.climate.copernicus.eu/api" 
CDS_KEY = "6a83cd0d-c34c-4fd1-ab54-24e2a59a163b" 

AREA = [40, 110, 20, 135]  # [N, W, S, E]
MAX_WORKERS = 5          # 并发数
# ========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_cds_config(url, key):
    if url: url = url.replace("url:", "").strip()
    if key: key = key.replace("key:", "").strip()
    return url, key

def get_typhoon_time_mapping(excel_path, sheets):
    print(f"正在读取 Excel: {excel_path}")
    mapping_list = []
    unique_times = set()
    try:
        xls = pd.ExcelFile(excel_path)
        for sheet in sheets:
            if sheet not in xls.sheet_names: continue
            df = pd.read_excel(xls, sheet_name=sheet)
            df.columns = [c.strip() for c in df.columns]
            required = ['中文名', '英文名', '出现时刻']
            if not all(col in df.columns for col in required): continue
            for idx, row in df.iterrows():
                try:
                    t_str = str(row['出现时刻'])
                    dt_utc = pd.to_datetime(t_str) - timedelta(hours=8)
                    unique_times.add(dt_utc)
                    mapping_list.append({
                        "sheet": sheet, "cn_name": str(row['中文名']).strip(),
                        "en_name": str(row['英文名']).strip(), "utc_time": dt_utc
                    })
                except: pass
    except Exception as e:
        print(f"[Error] 读取 Excel: {e}")
        return [], []
    return mapping_list, sorted(list(unique_times))

def process_single_time(dt, cache_dir, url, key):
    """单变量下载逻辑"""
    try:
        c = cdsapi.Client(url=url, key=key, quiet=True)
    except Exception as e:
        return dt, False, f"Client Init Failed: {e}"

    # 文件名
    fname = f"ERA5_500hPa_{dt.strftime('%Y%m%d_%H')}.nc"
    fpath = cache_dir / fname
    
    if fpath.exists():
        return dt, True, "已存在 (Skipped)"

    # 请求参数 (仅下载 500hPa 位势)
    request_params = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',  # 新版API必填
        'variable': 'geopotential',
        'pressure_level': '500',
        'year': dt.strftime("%Y"),
        'month': dt.strftime("%m"),
        'day': dt.strftime("%d"),
        'time': dt.strftime("%H:00"),
        'area': AREA,
    }

    try:
        c.retrieve('reanalysis-era5-pressure-levels', request_params, str(fpath))
        return dt, True, "下载成功"
    except Exception as e:
        # 失败时删除可能生成的空文件
        if fpath.exists(): 
            try: os.remove(fpath)
            except: pass
        return dt, False, str(e)

def main():
    clean_url, clean_key = clean_cds_config(CDS_URL, CDS_KEY)
    
    # 缓存目录 (仅存放单变量文件)
    CACHE_DIR = BASE_DIR / "Cache_Single_Files"
    FINAL_DIR = BASE_DIR / "Output_Typhoons_NC"
    
    ensure_dir(CACHE_DIR)
    
    mapping_list, unique_times = get_typhoon_time_mapping(INPUT_EXCEL, ["8-9级", "10级及以上"])
    
    if not unique_times: return

    total = len(unique_times)
    print(f"\n>>> 启动 [单变量] 并发下载 (Max Workers: {MAX_WORKERS})...")
    print("    目标: 500hPa Geopotential Only")
    
    failed_list = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_time = {
            executor.submit(process_single_time, dt, CACHE_DIR, clean_url, clean_key): dt 
            for dt in unique_times
        }
        
        count = 0
        for future in as_completed(future_to_time):
            count += 1
            dt, success, msg = future.result()
            symbol = "✅" if success else "❌"
            print(f"[{count}/{total}] {symbol} {dt} (UTC) -> {msg}")
            if not success: failed_list.append((dt, msg))

    print("\n" + "="*50)
    print(f"下载结束。成功: {total - len(failed_list)}, 失败: {len(failed_list)}")
    if failed_list:
        print("失败列表:")
        for ft in failed_list: print(f"  {ft[0]}: {ft[1]}")
    print("="*50)

    print(f"\n>>> 正在分发文件...")
    ensure_dir(FINAL_DIR)
    dist_count = 0
    for item in mapping_list:
        dt = item['utc_time']
        cache_name = f"ERA5_500hPa_{dt.strftime('%Y%m%d_%H')}.nc"
        cache_path = CACHE_DIR / cache_name
        
        if cache_path.exists():
            safe_cn = item['cn_name'].replace('/', '_')
            safe_en = item['en_name'].replace('/', '_')
            safe_sheet = item['sheet'].replace('/', '_')
            time_str = dt.strftime('%Y%m%d_%H')
            
            target_name = f"{safe_cn}_{safe_en}_{safe_sheet}_{time_str}_UTC_500hPa.nc"
            target_path = FINAL_DIR / target_name
            
            try:
                shutil.copy2(cache_path, target_path)
                dist_count += 1
            except: pass
            
    print(f"[Success] 已生成 {dist_count} 个台风文件！位于: {FINAL_DIR}")

if __name__ == "__main__":
    main()