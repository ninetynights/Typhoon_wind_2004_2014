import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
#               配置区域
# ==========================================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"

# 1. 最佳路径文件夹 (包含 CH2004BST.txt 等)
TRACK_DIR = os.path.join(BASE_DIR, "热带气旋最佳路径数据集")

# 2. 影响时间窗 Excel (用于筛选需要的时间段)
EXCEL_FILE = os.path.join(BASE_DIR, "数据", "2004_2024_影响台风_大风.xlsx")

# 3. 输出结果路径
OUTPUT_FILE = os.path.join(BASE_DIR, "输出_机器学习", "Typhoon_Tracks_Hourly_Cubic_Features_CubicSpline.csv")

# ==========================================
#           核心处理函数
# ==========================================

def read_excel_windows(path):
    """读取 Excel 获取每个台风的影响起止时间"""
    try:
        df = pd.read_excel(path)
        # 确保ID是4位字符串 (例如 401 -> 0401)
        df['中央台编号'] = df['中央台编号'].astype(str).str.strip().str.zfill(4)
        return df[['中央台编号', '大风开始时间', '大风结束时间']]
    except Exception as e:
        print(f"[错误] 读取Excel失败: {e}")
        return pd.DataFrame()

def read_track_file(filepath, target_tid):
    """
    读取单个年份的路径文件，提取指定台风的数据
    返回: DataFrame (包含 Time, Lat, Lon, Pressure, Wind_Speed_Center)
    """
    data = []
    current_id = None
    is_target = False
    
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = re.split(r'\s+', line)
            
            # 1. 识别头文件行 (66666)
            if line.startswith("66666"):
                if len(parts) >= 5:
                    # 获取台风编号 (第5列, 索引4)
                    current_id = parts[4].strip().zfill(4)
                    is_target = (current_id == target_tid)
                else:
                    is_target = False
            
            # 2. 读取数据行 (仅当是目标台风时)
            elif is_target:
                # 标准CMA BST格式: 
                # Col 0: 时间 (YYYYMMDDHH)
                # Col 2: 纬度 (0.1度)
                # Col 3: 经度 (0.1度)
                # Col 4: 中心气压 (hPa)
                # Col 5: 2分钟平均近中心最大风速 (m/s)
                
                if len(parts) >= 6:
                    try:
                        time_str = parts[0]
                        lat = float(parts[2]) / 10.0
                        lon = float(parts[3]) / 10.0
                        pressure = float(parts[4])
                        wind = float(parts[5]) 
                        
                        ts = pd.to_datetime(time_str, format="%Y%m%d%H")
                        
                        data.append({
                            'Time': ts,
                            'Lat': lat,
                            'Lon': lon,
                            'Pressure': pressure,
                            'Wind_Speed_Center': wind
                        })
                    except ValueError:
                        continue

    if not data:
        return None
    
    # 转为 DataFrame 并去重
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=['Time']).sort_values('Time')
    return df

def process_single_typhoon(tid, start_time, end_time, track_dir):
    """
    核心逻辑：读取 -> 截取 -> 插值 (路径用Cubic Spline，强度用Linear)
    """
    # 1. 确定文件路径
    try:
        y_short = int(tid[:2])
        year_full = 2000 + y_short if y_short <= 50 else 1900 + y_short
        txt_path = os.path.join(track_dir, f"CH{year_full}BST.txt")
    except:
        print(f"  [警告] 无法解析台风ID年份: {tid}")
        return None

    # 2. 读取原始数据
    df_raw = read_track_file(txt_path, tid)
    if df_raw is None or df_raw.empty:
        return None

    # 3. 设置时间索引
    df_raw.set_index('Time', inplace=True)
    
    # 4. 构建完整的小时级时间轴
    t_min = min(df_raw.index.min(), start_time)
    t_max = max(df_raw.index.max(), end_time)
    
    full_idx = pd.date_range(
        start=t_min.floor('H'), 
        end=t_max.ceil('H'), 
        freq='1H'
    )
    
    # 5. 重采样引入 NaN
    df_interp = df_raw.reindex(full_idx)
    
    # =========================================================
    #            【修改区域】 插值方法优化
    # =========================================================
    
    # A. 经纬度 (Lat, Lon) -> 使用 Cubic Spline (三次样条)
    # 注意：Spline 需要一定数量的数据点（通常>=4）。如果点太少会报错，需降级处理。
    try:
        # limit_direction='both' 允许在插值范围内填充首尾
        df_interp['Lat'] = df_interp['Lat'].interpolate(method='cubic', limit_direction='both')
        df_interp['Lon'] = df_interp['Lon'].interpolate(method='cubic', limit_direction='both')
    except Exception as e:
        # 如果数据点太少（例如只有2-3个记录），样条插值会失败，此时回退到线性插值
        # print(f"  [Info] 台风 {tid} 数据点不足以进行三次样条插值，降级为线性插值。")
        df_interp['Lat'] = df_interp['Lat'].interpolate(method='time', limit_direction='both')
        df_interp['Lon'] = df_interp['Lon'].interpolate(method='time', limit_direction='both')

    # B. 强度要素 (Pressure, Wind) -> 保持线性插值 (method='time')
    # 避免风速出现非物理的“过冲”现象
    df_interp['Pressure'] = df_interp['Pressure'].interpolate(method='time', limit_direction='both')
    df_interp['Wind_Speed_Center'] = df_interp['Wind_Speed_Center'].interpolate(method='time', limit_direction='both')
    
    # =========================================================

    # 6. 截取用户关注的“大风影响时段”
    mask = (df_interp.index >= start_time) & (df_interp.index <= end_time)
    df_final = df_interp[mask].copy()
    
    if df_final.empty:
        return None
        
    df_final.reset_index(inplace=True)
    df_final.rename(columns={'index': 'Time'}, inplace=True)
    df_final['TID'] = tid
    
    return df_final

# ==========================================
#               主程序
# ==========================================

def main():
    print(f"开始处理...")
    print(f"读取 Excel: {EXCEL_FILE}")
    df_excel = read_excel_windows(EXCEL_FILE)
    
    if df_excel.empty:
        print("[错误] Excel 数据为空或读取失败")
        return

    all_tracks = []
    total_count = len(df_excel)
    print(f"共需处理 {total_count} 个台风事件...")
    
    success_count = 0
    
    for idx, row in df_excel.iterrows():
        tid = row['中央台编号']
        t_start = row['大风开始时间']
        t_end = row['大风结束时间']
        
        if pd.isna(t_start) or pd.isna(t_end):
            continue
            
        print(f"[{idx+1}/{total_count}] 正在处理台风 {tid} ...", end="\r")
        
        df_res = process_single_typhoon(tid, t_start, t_end, TRACK_DIR)
        
        if df_res is not None:
            all_tracks.append(df_res)
            success_count += 1
            
    print(f"\n处理完成！成功提取 {success_count} 个台风的路径数据。")
    
    if all_tracks:
        print("正在合并数据...")
        final_df = pd.concat(all_tracks, ignore_index=True)
        
        cols = ['TID', 'Time', 'Lat', 'Lon', 'Pressure', 'Wind_Speed_Center']
        final_df = final_df[cols]
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"文件已保存: {OUTPUT_FILE}")
        print(f"总数据行数: {len(final_df)}")
        print("="*30)
        print("前5行预览:")
        print(final_df.head())
        print("="*30)
    else:
        print("警告：没有生成任何数据。")

if __name__ == "__main__":
    main()