"""
查看NC文件信息.py

功能：简单的打印 NetCDF 文件的维度、变量和属性信息。
"""

import xarray as xr
import netCDF4
from pathlib import Path

# ======= 请修改这里为您的实际文件路径 =======
# 您刚才提到的那个文件路径，大概在这个位置：
NC_FILE_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/ERA5_Data/Cache_Time_Files/ERA5_20040702_14.nc"
# ==========================================

def inspect_nc(path):
    fpath = Path(path)
    if not fpath.exists():
        print(f"[Error] 文件不存在: {fpath}")
        return

    print(f"\n{'='*20} 文件概览 {'='*20}")
    print(f"文件名: {fpath.name}")

    ds = xr.open_dataset(fpath)
    print("\n>>> Xarray 数据集详情:")
    print(ds)
        
    print("\n>>> 核心变量 'z' (位势) 的属性:")
    if 'z' in ds:
        print(ds['z'].attrs)
    elif 'geopotential' in ds:
        print(ds['geopotential'].attrs)

if __name__ == "__main__":
    inspect_nc(NC_FILE_PATH)