"""
ERA5_Download_V12_FullVariables_Merged.py

功能升级：
1. [多层合并] 每个时刻自动下载 "高空(PL)" 和 "地面(SL)" 两套数据，并合并为一个 NC 文件。
2. [变量扩充] 
   - 高空: 位势, U风, V风, 涡度, 散度 (Levels: 500, 850, 925)
   - 地面: 海平面气压, 10米U风, 10米V风
3. [输出] 生成包含全要素的台风独立文件，方便后续综合分析。

输入：AllTyphoons_Max_Summary.xlsx
输出：ERA5_Data_Full/Output_Typhoons_NC/
"""

import cdsapi
import pandas as pd
import xarray as xr
import shutil
import os
from pathlib import Path
from datetime import datetime, timedelta

# ======= 配置区域 =======

# 1. 输入 Excel 路径
INPUT_EXCEL = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分级统计/输出_台风过程分时段统计/AllTyphoons_Max_Summary.xlsx"

# 2. 根输出目录 (建议换个新目录，以免和旧数据混淆)
BASE_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/ERA5_Data")

# 3. CDS API 配置
CDS_URL = "https://cds.climate.copernicus.eu/api" 
CDS_KEY = "6a83cd0d-c34c-4fd1-ab54-24e2a59a163b"

# 4. 下载区域 [North, West, South, East]
AREA = [35, 115, 20, 130] 

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
            
            print(f"  -> 读取 Sheet: {sheet}")
            df = pd.read_excel(xls, sheet_name=sheet)
            df.columns = [c.strip() for c in df.columns]
            
            required = ['中文名', '英文名', '出现时刻']
            if not all(col in df.columns for col in required):
                print(f"    [Skip] {sheet} 缺少列")
                continue
            
            for idx, row in df.iterrows():
                try:
                    t_str = str(row['出现时刻'])
                    # CST -> UTC
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
    print(f"[Info] 需下载 {len(sorted_times)} 个时刻 (全要素合并)。")
    return mapping_list, sorted_times

def download_and_merge_times(time_list, cache_dir, url, key):
    # 初始化 Client
    if url and key:
        c = cdsapi.Client(url=url, key=key)
    else:
        try: c = cdsapi.Client()
        except: 
            print("[Fatal] API Key 未配置"); return

    ensure_dir(cache_dir)
    total = len(time_list)
    print(f"\n>>> 开始下载并合并 ({total} 个任务)...")
    
    # 定义两套请求参数
    # 1. 高空 (Pressure Levels)
    req_pl = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': [
            'geopotential', 
            'u_component_of_wind', 'v_component_of_wind',
            'vorticity', 'divergence'
        ],
        'pressure_level': ['500', '850', '925'], # 您指定的3层
        'year': '', 'month': '', 'day': '', 'time': '', # 动态填充
        'area': AREA,
    }
    
    # 2. 地面 (Single Levels)
    req_sl = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': [
            'mean_sea_level_pressure',
            '10m_u_component_of_wind', '10m_v_component_of_wind'
        ],
        'year': '', 'month': '', 'day': '', 'time': '',
        'area': AREA,
    }

    for i, dt in enumerate(time_list):
        # 最终合并文件
        fname_final = f"ERA5_Full_{dt.strftime('%Y%m%d_%H')}.nc"
        fpath_final = cache_dir / fname_final
        
        progress = f"[{i+1}/{total}]"
        
        if fpath_final.exists():
            print(f"  {progress} {fname_final} 已存在，跳过。")
            continue
            
        print(f"  {progress} 处理: {dt} (UTC)")
        
        # 临时文件路径
        fpath_pl = cache_dir / "temp_pl.nc"
        fpath_sl = cache_dir / "temp_sl.nc"
        
        # 更新时间参数
        t_params = {
            'year': dt.strftime("%Y"),
            'month': dt.strftime("%m"),
            'day': dt.strftime("%d"),
            'time': dt.strftime("%H:00")
        }
        req_pl.update(t_params)
        req_sl.update(t_params)
        
        try:
            # Step 1: 下载高空
            # print("    -> Downloading Pressure Levels...")
            c.retrieve('reanalysis-era5-pressure-levels', req_pl, str(fpath_pl))
            
            # Step 2: 下载地面
            # print("    -> Downloading Single Levels...")
            c.retrieve('reanalysis-era5-single-levels', req_sl, str(fpath_sl))
            
            # Step 3: 合并 (Merge)
            # print("    -> Merging...")
            try:
                ds_pl = xr.open_dataset(fpath_pl)
                ds_sl = xr.open_dataset(fpath_sl)
                
                # 合并两个数据集
                ds_merged = xr.merge([ds_pl, ds_sl])
                
                # 保存最终文件
                ds_merged.to_netcdf(fpath_final)
                
                # 关闭句柄
                ds_pl.close()
                ds_sl.close()
                ds_merged.close()
                
                # 删除临时文件
                if fpath_pl.exists(): os.remove(fpath_pl)
                if fpath_sl.exists(): os.remove(fpath_sl)
                
                print(f"    -> 成功生成合并文件")
                
            except Exception as e:
                print(f"    [Error] 合并失败 (Xarray): {e}")
                
        except Exception as e:
            print(f"    [Error] 下载或请求失败: {e}")

def distribute_files(mapping_list, cache_dir, output_dir):
    print(f"\n>>> 正在分发文件...")
    ensure_dir(output_dir)
    
    count = 0
    for item in mapping_list:
        dt = item['utc_time']
        cache_name = f"ERA5_Full_{dt.strftime('%Y%m%d_%H')}.nc"
        cache_path = cache_dir / cache_name
        
        if not cache_path.exists():
            continue
            
        safe_cn = item['cn_name'].replace('/', '_')
        safe_en = item['en_name'].replace('/', '_')
        safe_sheet = item['sheet'].replace('/', '_')
        time_str = dt.strftime('%Y%m%d_%H')
        
        # 加上 _Full 后缀表示包含全要素
        target_name = f"{safe_cn}_{safe_en}_{safe_sheet}_{time_str}_UTC_Full.nc"
        target_path = output_dir / target_name
        
        try:
            shutil.copy2(cache_path, target_path)
            count += 1
        except: pass
            
    print(f"[Success] 已生成 {count} 个全要素台风文件！位于: {output_dir}")

def main():
    clean_url, clean_key = clean_cds_config(CDS_URL, CDS_KEY)
    
    CACHE_DIR = BASE_DIR / "Cache_Full_Files"
    FINAL_DIR = BASE_DIR / "Output_Typhoons_NC"
    
    mapping_list, unique_times = get_typhoon_time_mapping(INPUT_EXCEL, ["8-9级", "10级及以上"])
    
    if unique_times:
        download_and_merge_times(unique_times, CACHE_DIR, clean_url, clean_key)
        distribute_files(mapping_list, CACHE_DIR, FINAL_DIR)

if __name__ == "__main__":
    main()