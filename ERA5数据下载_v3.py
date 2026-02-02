"""
ERA5_Download_V13_Parallel_FullVariables.py

功能：
1. [多线程并发] 同时处理多个时间点，榨干 CDS API 的允许配额，大幅缩短排队时间。
2. [全要素下载] 
   - Pressure Levels (500/850/925 hPa): 位势, U风, V风, 涡度, 散度
   - Single Levels: 海平面气压, 10米U风, 10米V风
3. [自动合并] 下载完成后，自动将 PL 和 SL 数据合并为一个 NC 文件。
4. [按台风归档] 生成的文件名格式：中文名_英文名_等级_时间_UTC_Full.nc

输入：AllTyphoons_Max_Summary.xlsx
输出：ERA5_Data_Full/Output_Typhoons_NC/
"""

import cdsapi
import pandas as pd
import xarray as xr
import shutil
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======= 配置区域 =======

# 1. 输入 Excel 路径
INPUT_EXCEL = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分级统计/输出_台风过程分时段统计/AllTyphoons_Max_Summary.xlsx"

# 2. 根输出目录
BASE_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/ERA5_Data_FULL")

# 3. CDS API 配置 (请务必填入您的 Key)
CDS_URL = "https://cds.climate.copernicus.eu/api" 
CDS_KEY = "6a83cd0d-c34c-4fd1-ab54-24e2a59a163b" 

# 4. 下载区域 [N, W, S, E]
AREA = [35, 115, 20, 130] 

# 5. 最大并发数 (建议 4~6，过高容易被 CDS 拒绝)
MAX_WORKERS = 5

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
            
            # 读取并清洗列名
            df = pd.read_excel(xls, sheet_name=sheet)
            df.columns = [c.strip() for c in df.columns]
            
            required = ['中文名', '英文名', '出现时刻']
            if not all(col in df.columns for col in required): continue
            
            for idx, row in df.iterrows():
                try:
                    t_str = str(row['出现时刻'])
                    # 北京时间 -> UTC (减8小时)
                    dt_utc = pd.to_datetime(t_str) - timedelta(hours=8)
                    
                    unique_times.add(dt_utc)
                    mapping_list.append({
                        "sheet": sheet,
                        "cn_name": str(row['中文名']).strip(),
                        "en_name": str(row['英文名']).strip(),
                        "utc_time": dt_utc
                    })
                except: pass
    except Exception as e:
        print(f"[Error] 读取 Excel: {e}")
        return [], []

    sorted_times = sorted(list(unique_times))
    print(f"[Info] 解析完毕。共需下载 {len(sorted_times)} 个唯一时刻 (全要素合并)。")
    return mapping_list, sorted_times

def process_single_time(dt, cache_dir, url, key):
    """
    单个时间点的处理逻辑：
    1. 构造请求 -> 2. 下载高空(PL) -> 3. 下载地面(SL) -> 4. 合并 -> 5. 清理临时文件
    """
    # 每个线程独立初始化 Client，避免冲突
    try:
        c = cdsapi.Client(url=url, key=key, quiet=True) # quiet=True 减少日志刷屏
    except Exception as e:
        return dt, False, f"Client Init Failed: {e}"

    # 最终文件名
    fname_final = f"ERA5_Full_{dt.strftime('%Y%m%d_%H')}.nc"
    fpath_final = cache_dir / fname_final
    
    if fpath_final.exists():
        return dt, True, "已存在 (Skipped)"

    # 定义请求参数
    t_params = {
        'year': dt.strftime("%Y"), 'month': dt.strftime("%m"),
        'day': dt.strftime("%d"), 'time': dt.strftime("%H:00"),
        'area': AREA
    }
    
    # 高空请求 (5层要素)
    req_pl = {
        'product_type': 'reanalysis', 'data_format': 'netcdf',
        'variable': ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vorticity', 'divergence'],
        'pressure_level': ['500', '850', '925'],
        **t_params
    }
    
    # 地面请求 (3要素)
    req_sl = {
        'product_type': 'reanalysis', 'data_format': 'netcdf',
        'variable': ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind'],
        **t_params
    }

    # 临时文件 (增加随机ID防止线程冲突)
    temp_id = f"{dt.strftime('%Y%m%d%H')}_{int(time.time()*1000)}"
    fpath_pl = cache_dir / f"temp_pl_{temp_id}.nc"
    fpath_sl = cache_dir / f"temp_sl_{temp_id}.nc"

    try:
        # Step 1: 下载
        c.retrieve('reanalysis-era5-pressure-levels', req_pl, str(fpath_pl))
        c.retrieve('reanalysis-era5-single-levels', req_sl, str(fpath_sl))
        
        # Step 2: 合并
        # 使用 xarray 合并两个文件
        ds_pl = xr.open_dataset(fpath_pl)
        ds_sl = xr.open_dataset(fpath_sl)
        
        ds_merged = xr.merge([ds_pl, ds_sl])
        
        # 保存并关闭
        ds_merged.to_netcdf(fpath_final)
        ds_pl.close(); ds_sl.close(); ds_merged.close()
        
        # Step 3: 清理临时文件
        if fpath_pl.exists(): os.remove(fpath_pl)
        if fpath_sl.exists(): os.remove(fpath_sl)
        
        return dt, True, "下载并合并成功"

    except Exception as e:
        # 失败时尝试清理残余
        try:
            if fpath_pl.exists(): os.remove(fpath_pl)
            if fpath_sl.exists(): os.remove(fpath_sl)
        except: pass
        return dt, False, str(e)

def main():
    clean_url, clean_key = clean_cds_config(CDS_URL, CDS_KEY)
    
    CACHE_DIR = BASE_DIR / "Cache_Full_Files"
    FINAL_DIR = BASE_DIR / "Output_Typhoons_NC"
    
    ensure_dir(CACHE_DIR)
    
    # 读取 Excel
    mapping_list, unique_times = get_typhoon_time_mapping(INPUT_EXCEL, ["8-9级", "10级及以上"])
    
    if not unique_times: return

    # --- 并发执行 ---
    total = len(unique_times)
    print(f"\n>>> 启动并发下载 (Max Workers: {MAX_WORKERS})...")
    print("    [提示] 多个任务将同时在后台排队，日志可能会交替刷新，请耐心等待。")
    
    failed_list = []
    
    # 使用线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_time = {
            executor.submit(process_single_time, dt, CACHE_DIR, clean_url, clean_key): dt 
            for dt in unique_times
        }
        
        count = 0
        for future in as_completed(future_to_time):
            count += 1
            dt, success, msg = future.result()
            
            status_symbol = "✅" if success else "❌"
            print(f"[{count}/{total}] {status_symbol} {dt} (UTC) -> {msg}")
            
            if not success:
                failed_list.append((dt, msg))

    # --- 报告 ---
    print("\n" + "="*50)
    print(f"下载结束。成功: {total - len(failed_list)}, 失败: {len(failed_list)}")
    if failed_list:
        print("失败列表 (请检查网络或稍后重试):")
        for ft in failed_list:
            print(f"  {ft[0]}: {ft[1]}")
    print("="*50)

    # --- 分发文件 ---
    print(f"\n>>> 正在生成台风独立文件...")
    ensure_dir(FINAL_DIR)
    
    dist_count = 0
    for item in mapping_list:
        dt = item['utc_time']
        cache_name = f"ERA5_Full_{dt.strftime('%Y%m%d_%H')}.nc"
        cache_path = CACHE_DIR / cache_name
        
        if cache_path.exists():
            safe_cn = item['cn_name'].replace('/', '_')
            safe_en = item['en_name'].replace('/', '_')
            safe_sheet = item['sheet'].replace('/', '_')
            time_str = dt.strftime('%Y%m%d_%H')
            
            target_name = f"{safe_cn}_{safe_en}_{safe_sheet}_{time_str}_UTC_Full.nc"
            target_path = FINAL_DIR / target_name
            
            try:
                shutil.copy2(cache_path, target_path)
                dist_count += 1
            except: pass
            
    print(f"[Success] 已生成 {dist_count} 个全要素台风文件！位于: {FINAL_DIR}")

if __name__ == "__main__":
    main()