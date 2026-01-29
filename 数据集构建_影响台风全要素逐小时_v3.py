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

# 1. 最佳路径文件夹
TRACK_DIR = os.path.join(BASE_DIR, "热带气旋最佳路径数据集")

# 2. 影响时间窗 Excel
EXCEL_FILE = os.path.join(BASE_DIR, "数据_v2", "2004_2024_影响台风_大风.xlsx")

# 3. 输出结果路径
OUTPUT_FILE = os.path.join(BASE_DIR, "输出_机器学习", "Typhoon_Tracks_Hourly_Cubic_Features_With_Velocity.csv")

# ==========================================
#           核心处理函数
# ==========================================

def read_excel_windows(path):
    """读取 Excel 获取每个台风的影响起止时间"""
    try:
        df = pd.read_excel(path)
        df['中央台编号'] = df['中央台编号'].astype(str).str.strip().str.zfill(4)
        return df[['中央台编号', '大风开始时间', '大风结束时间']]
    except Exception as e:
        print(f"[错误] 读取Excel失败: {e}")
        return pd.DataFrame()

def read_track_file(filepath, target_tid):
    """读取单个年份的路径文件"""
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
            
            if line.startswith("66666"):
                if len(parts) >= 5:
                    current_id = parts[4].strip().zfill(4)
                    is_target = (current_id == target_tid)
                else:
                    is_target = False
            
            elif is_target:
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
    
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=['Time']).sort_values('Time')
    return df

def calculate_movement(df):
    """
    【新增功能】计算台风移动速度(km/h)和移动方向(0-360度)
    输入: 包含 Lat, Lon 的 DataFrame
    输出: 增加了 Move_Speed, Move_Dir 列的 DataFrame
    """
    # 地球半径 (km)
    R = 6371.0
    
    # 1. 准备数据：将经纬度转换为弧度
    # shift(1) 获取上一时刻的经纬度
    lat1 = np.radians(df['Lat'].shift(1))
    lon1 = np.radians(df['Lon'].shift(1))
    lat2 = np.radians(df['Lat'])
    lon2 = np.radians(df['Lon'])
    
    # 2. 计算距离 (Haversine 公式)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    
    # 计算时间差 (小时)
    # 假设数据已经是每小时插值好的，但为了稳健性我们还是计算一下实际时间差
    time_diff = df.index.to_series().diff().dt.total_seconds() / 3600.0
    
    # 速度 = 距离 / 时间 (km/h)
    # 处理除零错误，虽然每小时数据一般不会是0
    speed = distance_km / time_diff
    
    # 3. 计算方向 (Bearing)
    # 公式: θ = atan2(sin(Δλ).cos(φ2), cos(φ1).sin(φ2) − sin(φ1).cos(φ2).cos(Δλ))
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(y, x))
    
    # 归一化为 0-360 度 (负数转正数)
    bearing = (bearing + 360) % 360
    
    # 将计算结果赋值回 DataFrame
    df['Move_Speed'] = speed
    df['Move_Dir'] = bearing
    
    # 第一行因为没有前一时刻数据，会是 NaN，我们可以用第二行的数据回填 (bfill)
    df['Move_Speed'] = df['Move_Speed'].fillna(method='bfill')
    df['Move_Dir'] = df['Move_Dir'].fillna(method='bfill')
    
    return df

def process_single_typhoon(tid, start_time, end_time, track_dir):
    try:
        y_short = int(tid[:2])
        year_full = 2000 + y_short if y_short <= 50 else 1900 + y_short
        txt_path = os.path.join(track_dir, f"CH{year_full}BST.txt")
    except:
        return None

    df_raw = read_track_file(txt_path, tid)
    if df_raw is None or df_raw.empty:
        return None

    df_raw.set_index('Time', inplace=True)
    
    t_min = min(df_raw.index.min(), start_time)
    t_max = max(df_raw.index.max(), end_time)
    
    full_idx = pd.date_range(start=t_min.floor('H'), end=t_max.ceil('H'), freq='1H')
    df_interp = df_raw.reindex(full_idx)
    
    # --- 插值 ---
    try:
        df_interp['Lat'] = df_interp['Lat'].interpolate(method='cubic', limit_direction='both')
        df_interp['Lon'] = df_interp['Lon'].interpolate(method='cubic', limit_direction='both')
    except:
        df_interp['Lat'] = df_interp['Lat'].interpolate(method='time', limit_direction='both')
        df_interp['Lon'] = df_interp['Lon'].interpolate(method='time', limit_direction='both')

    df_interp['Pressure'] = df_interp['Pressure'].interpolate(method='time', limit_direction='both')
    df_interp['Wind_Speed_Center'] = df_interp['Wind_Speed_Center'].interpolate(method='time', limit_direction='both')
    
    # =========================================================
    #            【新增步骤】 计算移动速度和方向
    # =========================================================
    # 务必在截取时间窗之前计算，以保证第一行数据的准确性
    df_interp = calculate_movement(df_interp)
    # =========================================================

    # 截取用户关注的时段
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
    df_excel = read_excel_windows(EXCEL_FILE)
    
    if df_excel.empty:
        print("[错误] Excel 数据为空")
        return

    all_tracks = []
    total_count = len(df_excel)
    
    for idx, row in df_excel.iterrows():
        tid = row['中央台编号']
        t_start = row['大风开始时间']
        t_end = row['大风结束时间']
        
        if pd.isna(t_start) or pd.isna(t_end):
            continue
            
        print(f"[{idx+1}/{total_count}] 处理台风 {tid} ...", end="\r")
        df_res = process_single_typhoon(tid, t_start, t_end, TRACK_DIR)
        
        if df_res is not None:
            all_tracks.append(df_res)
            
    print(f"\n处理完成。")
    
    if all_tracks:
        final_df = pd.concat(all_tracks, ignore_index=True)
        
        # 【修改】 在输出列中增加新特征
        cols = ['TID', 'Time', 'Lat', 'Lon', 'Pressure', 'Wind_Speed_Center', 'Move_Speed', 'Move_Dir']
        final_df = final_df[cols]
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"文件已保存: {OUTPUT_FILE}")
        print(final_df.head())
    else:
        print("警告：没有生成任何数据。")

if __name__ == "__main__":
    main()