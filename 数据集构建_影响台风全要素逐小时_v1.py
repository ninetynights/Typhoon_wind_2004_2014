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
OUTPUT_FILE = os.path.join(BASE_DIR, "输出_机器学习", "Typhoon_Tracks_Hourly_Full_Features.csv")

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
                # Col 1: 强度等级 ID
                # Col 2: 纬度 (0.1度)
                # Col 3: 经度 (0.1度)
                # Col 4: 中心气压 (hPa)
                # Col 5: 2分钟平均近中心最大风速 (m/s)
                
                # 例如: 2024052400 1  83 1283 1004      13
                if len(parts) >= 6:
                    try:
                        time_str = parts[0]
                        # 转换经纬度 (原始单位为0.1度)
                        lat = float(parts[2]) / 10.0
                        lon = float(parts[3]) / 10.0
                        pressure = float(parts[4])
                        wind = float(parts[5]) #这就是最大风速
                        
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
    # 按时间去重并排序，防止原始数据有重复行
    df = df.drop_duplicates(subset=['Time']).sort_values('Time')
    return df

def process_single_typhoon(tid, start_time, end_time, track_dir):
    """
    核心逻辑：读取 -> 截取 -> 插值
    """
    # 1. 确定文件路径 (根据 TID 前两位推断年份)
    # 0401 -> 2004, 1901 -> 2019, 2301 -> 2023, 2401 -> 2024
    try:
        y_short = int(tid[:2])
        # 简单判断：如果小于50认为是2000后，否则是1900后
        year_full = 2000 + y_short if y_short <= 50 else 1900 + y_short
        txt_path = os.path.join(track_dir, f"CH{year_full}BST.txt")
    except:
        print(f"  [警告] 无法解析台风ID年份: {tid}")
        return None

    # 2. 读取原始数据
    df_raw = read_track_file(txt_path, tid)
    if df_raw is None or df_raw.empty:
        # 尝试打印路径以便调试
        # print(f"  [提示] 文件中未找到台风 {tid} 的数据 (路径: {txt_path})")
        return None

    # 3. 设置时间索引
    df_raw.set_index('Time', inplace=True)
    
    # 4. 【关键步骤】构建完整的小时级时间轴
    # 扩展时间轴以包含 start_time 和 end_time (取整到小时)
    # 使用 floor/ceil 确保覆盖整个影响时段
    
    # 确保时间轴范围有效
    t_min = min(df_raw.index.min(), start_time)
    t_max = max(df_raw.index.max(), end_time)
    
    # 只在原始数据时间范围内进行安全插值，或者允许一定程度的外推(但不建议)
    # 这里的策略是：以原始数据的起止时间为基础，结合影响时段。
    # 但如果影响时段超出了台风记录的时段，插值会产生 NaN。
    # 我们先生成时间轴，后续截取。
    
    full_idx = pd.date_range(
        start=t_min.floor('H'), 
        end=t_max.ceil('H'), 
        freq='1H'
    )
    
    # 5. 【关键步骤】插值 (Interpolation)
    # reindex 会引入 NaN 行
    df_interp = df_raw.reindex(full_idx)
    
    # 使用 'time' 方法进行插值，这是处理非等间距时间序列最准确的方法
    # limit_direction='both' 允许填充首尾（如果在影响时段内）
    df_interp['Lat'] = df_interp['Lat'].interpolate(method='time')
    df_interp['Lon'] = df_interp['Lon'].interpolate(method='time')
    df_interp['Pressure'] = df_interp['Pressure'].interpolate(method='time')
    df_interp['Wind_Speed_Center'] = df_interp['Wind_Speed_Center'].interpolate(method='time')
    
    # 6. 截取用户关注的“大风影响时段”
    mask = (df_interp.index >= start_time) & (df_interp.index <= end_time)
    df_final = df_interp[mask].copy()
    
    # 如果截取后为空（比如影响时段完全在台风记录之外），则返回None
    if df_final.empty:
        return None
        
    # 重置索引，将 Time 变回一列
    df_final.reset_index(inplace=True)
    df_final.rename(columns={'index': 'Time'}, inplace=True)
    
    # 添加 TID 列
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
        
        # 简单校验时间有效性
        if pd.isna(t_start) or pd.isna(t_end):
            continue
            
        # 动态打印进度
        print(f"[{idx+1}/{total_count}] 正在处理台风 {tid} ...", end="\r")
        
        df_res = process_single_typhoon(tid, t_start, t_end, TRACK_DIR)
        
        if df_res is not None:
            all_tracks.append(df_res)
            success_count += 1
            
    print(f"\n处理完成！成功提取 {success_count} 个台风的路径数据。")
    
    if all_tracks:
        print("正在合并数据...")
        final_df = pd.concat(all_tracks, ignore_index=True)
        
        # 调整列顺序
        cols = ['TID', 'Time', 'Lat', 'Lon', 'Pressure', 'Wind_Speed_Center']
        final_df = final_df[cols]
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"文件已保存: {OUTPUT_FILE}")
        print(f"总数据行数: {len(final_df)}")
        print("="*30)
        print("前5行预览 (检查 Wind_Speed_Center 是否有值):")
        print(final_df.head())
        print("="*30)
    else:
        print("警告：没有生成任何数据，请检查：")
        print("1. 路径文件夹配置是否正确")
        print("2. Excel 中的台风编号年份是否与 txt 文件名匹配")

if __name__ == "__main__":
    main()