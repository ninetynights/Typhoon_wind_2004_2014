"""
统计_台风过程分时段大风频次_V8_Excel自动适配版.py

修复说明:
1. [核心修复] `load_typhoon_time_table` 函数现在会自动判断文件后缀。
   - 如果是 .xlsx，自动调用 pd.read_excel。
   - 如果是 .csv，调用 pd.read_csv。
2. [路径更新] 已将时间表路径更新为您报错信息中的实际路径。

输入：NC_PATH, TIME_TABLE_PATH (.xlsx 或 .csv 均可)
输出：输出_台风过程分时段统计/
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

# ======= 配置区域 =======
# 1. NetCDF 数据路径
NC_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据_v2/Refined_Combine_Stations_ExMaxWind+SLP+StP_Fixed_2004_2024.nc"

# 2. 台风时间表路径 (根据您报错信息更新)
TIME_TABLE_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据_v2/2004_2024_影响台风_大风.xlsx"

# 3. 输出目录
BASE_OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_大风分级统计")

# --- 风级定义 ---
LEVELS_EXACT = {
    "8级":  (17.2, 20.7),
    "9级":  (20.8, 24.4),
    "10级": (24.5, 28.4),
    "11级": (28.5, 32.6),
    "12级": (32.7, 36.9),
    "13级": (37.0, 41.4),
    "14级": (41.5, 46.1)
}

LEVELS_RANGE = {
    "8-9级":  (17.2, 24.4),
    "8-10级": (17.2, 28.4),
    "10-11级": (24.5, 32.6)
}

LEVELS_EXCEED = {
    "8级及以上": 17.2,   # 新增
    "9级及以上": 20.8,
    "10级及以上": 24.5,
    "11级及以上": 28.5,
    "12级及以上": 32.7
}
# ================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\\s]+", "_", name).strip("_")

def parse_mapping(attr_str: str):
    if not attr_str:
        return {}
    pairs = (p for p in attr_str.strip().split(";") if ":" in p)
    return {k.strip(): v.strip() for k, v in (q.split(":", 1) for q in pairs)}

def load_typhoon_time_table(file_path):
    """
    智能读取时间表：支持 .xlsx 和 .csv
    """
    path_str = str(file_path)
    print(f"\n[TimeTable] 正在读取台风时间表: {path_str}")
    
    try:
        # 自动判断文件类型
        if path_str.endswith('.xlsx') or path_str.endswith('.xls'):
            print("  -> 检测到 Excel 格式，使用 read_excel...")
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            print("  -> 检测到 CSV 格式，使用 read_csv...")
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("  -> UTF-8 解码失败，尝试 GBK...")
                df = pd.read_csv(file_path, encoding='gbk')

        # 清理列名空格
        df.columns = [str(c).strip() for c in df.columns]
        
        # 必须包含的列
        required = ['中央台编号', '年份+序号', '中文名称', '大风开始时间']
        for col in required:
            if col not in df.columns:
                print(f"[Error] 时间表缺少列: {col}。检测到的列名: {df.columns.tolist()}")
                return {}, {}, {}

        # 确保时间列格式正确
        df['大风开始时间'] = pd.to_datetime(df['大风开始时间'], errors='coerce')
        
        # 移除无效时间行
        valid_rows = df.dropna(subset=['大风开始时间'])
        dropped_count = len(df) - len(valid_rows)
        if dropped_count > 0:
            print(f"  -> 已过滤 {dropped_count} 行无效时间记录。")
        
        df = valid_rows

        # 构建映射字典
        # 1. 中央台编号
        map_id = dict(zip(df['中央台编号'].astype(str).str.strip(), df['大风开始时间']))
        # 2. 年份序号
        map_seq = dict(zip(df['年份+序号'].astype(str).str.strip(), df['大风开始时间']))
        # 3. 中文名称
        map_name = dict(zip(df['中文名称'].astype(str).str.strip(), df['大风开始时间']))
        
        print(f"[TimeTable] 加载成功。有效记录 {len(df)} 条。")
        return map_id, map_seq, map_name

    except Exception as e:
        print(f"[Error] 读取时间表失败: {e}")
        return {}, {}, {}

def get_fallback_time_strings(nc_obj, n_time):
    """备选时间解析"""
    vars_dict = nc_obj.variables
    time_strs = [None] * n_time
    try:
        if 'TL' in vars_dict:
            raw = vars_dict['TL'][:]
            for i in range(n_time):
                val = raw[i]
                if isinstance(val, bytes): val = val.decode('utf-8')
                s = str(val).strip()
                if len(s) == 10: time_strs[i] = f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:00"
                else: time_strs[i] = s
            return time_strs
        elif 'INITTIME' in vars_dict:
            raw = vars_dict['INITTIME'][:]
            for i in range(n_time):
                val = raw[i]
                if isinstance(val, bytes): val = val.decode('utf-8')
                s = str(val).strip()
                if len(s) == 10: time_strs[i] = f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:00"
                else: time_strs[i] = s
            return time_strs
    except: pass
    return [f"TimeIndex_{i}" for i in range(n_time)]


def main():
    # 1. 加载时间表
    map_id, map_seq, map_name = load_typhoon_time_table(TIME_TABLE_PATH)
    
    print(f"\n正在读取 NetCDF 文件: {NC_PATH}")
    nc = Dataset(NC_PATH, 'r')

    id_to_index = parse_mapping(getattr(nc, 'id_to_index', None))
    index_to_cn = parse_mapping(getattr(nc, 'index_to_cn', None))
    index_to_en = parse_mapping(getattr(nc, 'index_to_en', None))

    print("正在加载风速和台风ID数据...")
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :])
    ty_ids = np.array(nc.variables['typhoon_id_index'][:, 0, :])
    
    if np.issubdtype(ty_ids.dtype, np.floating):
        ty_ids_int = np.full_like(ty_ids, fill_value=-1, dtype=int)
        valid = ~np.isnan(ty_ids)
        ty_ids_int[valid] = ty_ids[valid].astype(int)
        ty_ids = ty_ids_int
    else:
        ty_ids = ty_ids.astype(int)

    n_time, n_sta = wind_speeds.shape
    fallback_time_strs = get_fallback_time_strings(nc, n_time)

    out_dir = BASE_OUTPUT_DIR / "输出_台风过程分时段统计（全）"
    csv_dir = out_dir / "csv"
    ensure_dir(csv_dir)

    if id_to_index:
        items = []
        for tid_str, idx_str in id_to_index.items():
            try: items.append((tid_str, int(str(idx_str).strip())))
            except: pass
        items.sort(key=lambda x: x[0])
    else:
        uniq = sorted({int(x) for x in np.unique(ty_ids) if int(x) >= 0})
        items = [(str(idx), idx) for idx in uniq]

    all_levels = list(LEVELS_EXACT.keys()) + list(LEVELS_RANGE.keys()) + list(LEVELS_EXCEED.keys())
    summary_collectors = {lvl: [] for lvl in all_levels}

    print("\n" + "="*80)
    print("开始分台风处理 (时间计算: 表格 StartTime + 相对小时)...")
    print("="*80)

    for tid_str, ty_idx in items:
        cn_name = index_to_cn.get(str(ty_idx), "")
        en_name = index_to_en.get(str(ty_idx), "")
        ty_label = f"{tid_str} {cn_name} ({en_name})"
        
        active_time_mask = np.any(ty_ids == ty_idx, axis=1)
        active_time_indices = np.where(active_time_mask)[0]

        if len(active_time_indices) == 0:
            continue
            
        # --- 匹配逻辑 ---
        start_dt = None
        match_source = ""
        
        if tid_str in map_id:
            start_dt = map_id[tid_str]
            match_source = f"ID匹配({tid_str})"
        elif tid_str in map_seq:
            start_dt = map_seq[tid_str]
            match_source = f"序号匹配({tid_str})"
        elif cn_name in map_name:
            start_dt = map_name[cn_name]
            match_source = f"中文名匹配({cn_name})"
            
        if start_dt is not None:
            print(f"\n>>> 正在分析台风: {ty_label}")
            print(f"    [时间锁定] {match_source} -> 起始时间: {start_dt}")
        else:
            print(f"\n>>> 正在分析台风: {ty_label}")
            print(f"    [Warn] 未在时间表中找到对应记录，使用 NC 备选时间。")

        records = []
        for step_i, t_idx in enumerate(active_time_indices):
            # 计算时间
            if start_dt is not None:
                current_dt = start_dt + pd.Timedelta(hours=step_i)
                time_str = current_dt.strftime("%Y-%m-%d %H:00")
            else:
                time_str = fallback_time_strs[t_idx]
            
            w_t = wind_speeds[t_idx, :]
            id_t = ty_ids[t_idx, :]
            mask_ty = (id_t == ty_idx)
            w_active = w_t[mask_ty]
            
            row_data = {
                "Time_String": time_str,
                "Relative_Hour": step_i + 1,
                "Active_Station_Count": len(w_active)
            }

            for name, (low, high) in LEVELS_EXACT.items():
                row_data[name] = np.sum((w_active >= low) & (w_active <= high))
            for name, (low, high) in LEVELS_RANGE.items():
                row_data[name] = np.sum((w_active >= low) & (w_active <= high))
            for name, thresh in LEVELS_EXCEED.items():
                row_data[name] = np.sum(w_active >= thresh)
            
            records.append(row_data)

        df = pd.DataFrame(records)
        cols = list(df.columns)
        if "Time_String" in cols:
            cols.insert(0, cols.pop(cols.index("Time_String")))
            df = df[cols]
            
        csv_name = f"TimeSeries_{sanitize_filename(tid_str)}_{sanitize_filename(cn_name)}.csv"
        df.to_csv(csv_dir / csv_name, index=False, encoding="utf-8-sig")

        print(f"    [统计摘要 - 站点数最多时刻]")
        for lvl_name in all_levels:
            if lvl_name not in df.columns: continue

            max_idx = df[lvl_name].idxmax()
            max_row = df.loc[max_idx]
            max_val = max_row[lvl_name]
            peak_time = max_row['Time_String']
            
            if max_val > 0:
                summary_collectors[lvl_name].append({
                    "台风编号": tid_str,
                    "中文名": cn_name,
                    "英文名": en_name,
                    "最大站点数": max_val,
                    "出现时刻": peak_time,
                    "相对小时": max_row['Relative_Hour']
                })
                #print(f"      {lvl_name:<10}: {max_val:3d} 站 @ {peak_time}")

    # 生成 Excel
    excel_path = out_dir / "AllTyphoons_Max_Summary.xlsx"
    print(f"\n正在生成汇总 Excel 文件: {excel_path} ...")
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for category, level_dict in [("单级别", LEVELS_EXACT), ("组合级", LEVELS_RANGE), ("阈值级", LEVELS_EXCEED)]:
                for lvl_name in level_dict.keys():
                    data = summary_collectors.get(lvl_name, [])
                    if data:
                        pd.DataFrame(data).to_excel(writer, sheet_name=lvl_name, index=False)
        print(f"[OK] 汇总 Excel 已保存。")
    except Exception as e:
        print(f"[Error] 保存 Excel 失败: {e}")
        summary_dir = out_dir / "summary_sheets"
        ensure_dir(summary_dir)
        for lvl_name, data in summary_collectors.items():
            if data:
                pd.DataFrame(data).to_csv(summary_dir / f"Summary_{lvl_name}.csv", index=False, encoding="utf-8-sig")

    print("\n" + "="*80)
    print("统计完成。")
    print("="*80)
    nc.close()

if __name__ == "__main__":
    main()