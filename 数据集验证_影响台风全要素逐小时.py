import os
import pandas as pd
import numpy as np

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"

# 1. 刚刚生成的逐小时插值文件
TRACK_FILE = os.path.join(BASE_DIR, "输出_机器学习", "Typhoon_Tracks_Hourly_Full_Features.csv")

# 2. 原始的时间窗 Excel
EXCEL_FILE = os.path.join(BASE_DIR, "数据", "2004_2024_影响台风_大风.xlsx")

# ================= 验证逻辑 =================

def verify_data():
    print(">>> 开始核查台风路径插值数据...\n")
    
    # 1. 读取数据
    if not os.path.exists(TRACK_FILE):
        print(f"[错误] 找不到插值文件: {TRACK_FILE}")
        return
    
    print(f"正在读取插值结果: {TRACK_FILE}")
    df_track = pd.read_csv(TRACK_FILE, dtype={'TID': str})
    df_track['Time'] = pd.to_datetime(df_track['Time'])
    
    print(f"正在读取参考 Excel: {EXCEL_FILE}")
    try:
        df_excel = pd.read_excel(EXCEL_FILE)
        df_excel['中央台编号'] = df_excel['中央台编号'].astype(str).str.strip().str.zfill(4)
        df_excel['大风开始时间'] = pd.to_datetime(df_excel['大风开始时间'])
        df_excel['大风结束时间'] = pd.to_datetime(df_excel['大风结束时间'])
    except Exception as e:
        print(f"[错误] Excel 读取失败: {e}")
        return

    # 2. 统计变量
    total_events = len(df_excel)
    pass_count = 0
    fail_count = 0
    missing_tids = []
    
    print(f"\n{'='*60}")
    print(f"{'台风编号':<8} | {'覆盖情况':<10} | {'连续性':<10} | {'数值完整性':<10} | {'状态'}")
    print(f"{'-'*60}")

    # 3. 逐个台风核查
    for _, row in df_excel.iterrows():
        tid = row['中央台编号']
        t_start = row['大风开始时间']
        t_end = row['大风结束时间']
        
        # 3.1 检查是否存在
        track_subset = df_track[df_track['TID'] == tid]
        
        if track_subset.empty:
            print(f"{tid:<12} | {'缺失':<14} | {'-':<13} | {'-':<13} | ❌ 缺失")
            fail_count += 1
            missing_tids.append(tid)
            continue
            
        # 3.2 检查时间覆盖 (Allow 1 hour tolerance)
        # 插值数据的最小时间应该 <= 开始时间，最大时间 >= 结束时间
        # 注意：由于插值是按整点进行的，如果大风开始时间是 09:30，插值数据可能有 09:00 或 10:00
        # 这里我们检查 track_min <= t_start 和 track_max >= t_end
        
        t_track_min = track_subset['Time'].min()
        t_track_max = track_subset['Time'].max()
        
        # 宽松检查：允许 1 小时的误差（因为 floor/ceil）
        coverage_ok = (t_track_min <= t_start + pd.Timedelta(hours=1)) and \
                      (t_track_max >= t_end - pd.Timedelta(hours=1))
        
        cov_status = "完整" if coverage_ok else "不全"
        if not coverage_ok:
            # 打印具体缺了多少
            msg = ""
            if t_track_min > t_start: msg += "缺头 "
            if t_track_max < t_end: msg += "少尾"
            cov_status = msg
            
        # 3.3 检查连续性 (是否每小时一条)
        # 计算时间差
        track_subset = track_subset.sort_values('Time')
        time_diffs = track_subset['Time'].diff().dropna()
        # 检查是否所有间隔都是 1 小时
        is_continuous = np.all(time_diffs == pd.Timedelta(hours=1))
        cont_status = "OK" if is_continuous else "断裂"
        
        # 3.4 检查数值 (NaN)
        # 检查 Lat, Lon, Pressure, Wind_Speed_Center 是否有空值
        cols_to_check = ['Lat', 'Lon', 'Pressure', 'Wind_Speed_Center']
        has_nan = track_subset[cols_to_check].isnull().values.any()
        val_status = "OK" if not has_nan else "含空值"
        
        # 3.5 综合判定
        if coverage_ok and is_continuous and not has_nan:
            status = "✅ 通过"
            pass_count += 1
        else:
            status = "⚠️ 异常"
            fail_count += 1
            
        print(f"{tid:<12} | {cov_status:<14} | {cont_status:<13} | {val_status:<13} | {status}")

    # 4. 总结
    print(f"{'='*60}")
    print(f"\n>>> 核查总结")
    print(f"总事件数: {total_events}")
    print(f"通过: {pass_count}")
    print(f"失败/异常: {fail_count}")
    
    if missing_tids:
        print(f"\n[注意] 以下台风在插值文件中完全缺失 (可能是路径源文件缺失):")
        print(", ".join(missing_tids))
        
    print(f"\n文件路径: {TRACK_FILE}")

if __name__ == "__main__":
    verify_data()