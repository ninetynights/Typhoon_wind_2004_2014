"""
查看_站点检验_含气压.py — 单站点台风影响时序快速可视化（含风速、风向、气压）

目的与功能：
- 快速检查并可视化指定台风在指定观测站的逐小时风速、风向、海平面气压与站点气压。
- 读取 NetCDF 文件：All_Typhoons_ExMaxWind+SLP+StP_2004_2024.nc

主要修改点：
- 新增读取变量：sea_level_pressure (SLP), station_pressure (StP)
- 新增双Y轴绘图：左轴显示风速，右轴显示气压
- 底部统计信息增加气压极值
- [新增] 在终端打印前5个时刻的详细数据

输入：
- target_typhoon_id: 台风编号
- target_station_id: 站点编号
- nc_path: NetCDF 文件路径
"""

from netCDF4 import Dataset, num2date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import re

# 设置字体以支持中文显示 (根据系统调整，MacOS常为Heiti TC，Windows常为SimHei)
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 用户输入设置 ---
target_typhoon_id = '2421'   # 示例台风编号
target_station_id = '58446'  # 示例站点编号
# [修改点1] 更新为包含气压数据的新文件路径
# nc_path = '/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据_v2/All_Typhoons_ExMaxWind+SLP+StP_2004_2024.nc'
nc_path = '/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据_v2/Representative_Stations_Typhoons_ExMaxWind+SLP+StP_Fixed_2004_2024.nc'

# --------------------

try:
    # 打开 NetCDF 文件
    nc = Dataset(nc_path)
except FileNotFoundError:
    print(f"错误：找不到文件 {nc_path}")
    exit()

# 解析映射属性
def parse_mapping(attr_str):
    pairs = attr_str.strip().split(";")
    return {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in pairs if ":" in p)}

try:
    id_to_index = parse_mapping(nc.getncattr('id_to_index'))
    index_to_cn = parse_mapping(nc.getncattr('index_to_cn'))
    index_to_en = parse_mapping(nc.getncattr('index_to_en'))
except Exception as e:
    print("属性解析失败，请检查NC文件属性格式:", e)
    exit()

# 获取台风索引及名称
if target_typhoon_id not in id_to_index:
    raise ValueError(f"台风编号 {target_typhoon_id} 不在数据中")
typhoon_index = int(id_to_index[target_typhoon_id])

cn_name = index_to_cn.get(str(typhoon_index), "Unknown")
en_name = index_to_en.get(str(typhoon_index), "Unknown")

# 查找站点索引
stid_list = nc.variables['STID'][:].astype(str)
# 去除可能存在的空格
stid_list = np.char.strip(stid_list)

if target_station_id not in stid_list:
    raise ValueError(f"站点编号 {target_station_id} 不在数据中")
station_index = np.where(stid_list == target_station_id)[0][0]

# --- [修改点2] 提取数据 (新增气压变量) ---
typhoon_ids = nc.variables['typhoon_id_index'][:, 0, station_index]
wind_speeds = nc.variables['wind_velocity'][:, 0, station_index]
wind_dirs = nc.variables['wind_direction'][:, 0, station_index]

# 尝试读取气压变量，如果不存在则报错或给全NaN
try:
    slp_vals = nc.variables['sea_level_pressure'][:, 0, station_index]
except KeyError:
    print("警告：文件中未找到 'sea_level_pressure' 变量")
    slp_vals = np.full_like(wind_speeds, np.nan)

try:
    stp_vals = nc.variables['station_pressure'][:, 0, station_index]
except KeyError:
    print("警告：文件中未找到 'station_pressure' 变量")
    stp_vals = np.full_like(wind_speeds, np.nan)

init_times = nc.variables['INITTIME'][:]
init_units = nc.variables['INITTIME'].units
time_dt = num2date(init_times, init_units)

# 找出台风影响时间段索引
valid_idx = np.where(typhoon_ids == typhoon_index)[0]
if len(valid_idx) == 0:
    print(f"提示：站点 {target_station_id} 在台风 {target_typhoon_id} 期间没有被标记为影响时段。")
    # 可以选择退出或继续（这里选择退出）
    exit()

# 数据切片
times = [datetime.datetime(t.year, t.month, t.day, t.hour, t.minute) for t in time_dt[valid_idx]]
ws = wind_speeds[valid_idx]
wd = wind_dirs[valid_idx]
slp = slp_vals[valid_idx]
stp = stp_vals[valid_idx]

# --- [修改点7] 终端打印前5个时间点数据 ---
print(f"\n--- 台风 {target_typhoon_id} ({cn_name}) @ 站点 {target_station_id} ---")
print(f"{'Time':<20} | {'Wind(m/s)':<10} | {'SLP(hPa)':<10} | {'StP(hPa)':<10}")
print("-" * 60)
for i in range(min(5, len(times))):
    t_str = times[i].strftime('%Y-%m-%d %H:%M')
    ws_val = f"{ws[i]:.1f}" if not np.isnan(ws[i]) else "NaN"
    slp_val = f"{slp[i]:.1f}" if not np.isnan(slp[i]) else "NaN"
    stp_val = f"{stp[i]:.1f}" if not np.isnan(stp[i]) else "NaN"
    print(f"{t_str:<20} | {ws_val:<10} | {slp_val:<10} | {stp_val:<10}")
print("-" * 60 + "\n")
# ---------------------------------------

# 计算风速方向分量（单位向量 * 风速，用于风向杆）
# 注意：barbs 需要 U, V 分量。气象定义中 风向是从哪吹来。
# U = -ws * sin(wd), V = -ws * cos(wd) 是将风向转为UV矢量方向
theta_rad = np.deg2rad(wd)
U = -ws * np.sin(theta_rad)
V = -ws * np.cos(theta_rad)

# 计算统计指标
avg_ws = np.nanmean(ws)
max_ws = np.nanmax(ws)
min_slp = np.nanmin(slp)
min_stp = np.nanmin(stp)
duration_hr = len(times)

# --- [修改点3] 可视化 (双轴绘制) ---
fig, ax1 = plt.subplots(figsize=(14, 7))

# 1. 绘制左轴：风速 (Wind Speed)
# zorder控制绘图层级，值越大越靠上
l1, = ax1.plot(times, ws, color='darkgrey', label='Max Wind Speed (m/s)', linewidth=1.2, zorder=3)

# 标注每个点的风速数值
for x, y in zip(times, ws):
    if not np.isnan(y):
        ax1.text(x, y - 0.2, f"{y:.1f}", ha='center', va='top', fontsize=8, color='grey', zorder=5)

# 绘制风向杆
ax1.barbs(times, ws, U, V, length=6, color='black', linewidth=1,
          barb_increments=dict(half=2, full=4, flag=20), zorder=4)

ax1.set_xlabel('Time (Hourly)')
ax1.set_ylabel('Wind Speed (m/s)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle='--', color='gray', alpha=0.3)

# 2. 绘制右轴：气压 (Pressure)
ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴

# 绘制海平面气压 (SLP)
l2, = ax2.plot(times, slp, color='royalblue', label='Sea Level Pressure (hPa)', linewidth=1.5, linestyle='-', alpha=0.9, zorder=2)
# 绘制站点气压 (StP)
l3, = ax2.plot(times, stp, color='forestgreen', label='Station Pressure (hPa)', linewidth=1.5, linestyle='--', alpha=0.8, zorder=2)

ax2.set_ylabel('Pressure (hPa)', color='royalblue')
ax2.tick_params(axis='y', labelcolor='royalblue')

# 标题设置
ax1.set_title(f"Typhoon {target_typhoon_id} - {cn_name} ({en_name}) - Station {target_station_id}\n"
              f"Hourly Obs: Wind & Pressure", fontsize=14, pad=15)

# --- [修改点4] 格式化时间轴 ---
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 根据时长可自动调整，这里设为1小时
fig.autofmt_xdate(rotation=45)

# --- [修改点5] 合并图例 ---
# 获取ax1和ax2的图例句柄和标签
lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)

# --- [修改点6] 底部统计信息 ---
# 只有在非全是NaN的情况下才显示数值
min_slp_str = f"{min_slp:.1f}" if not np.isnan(min_slp) else "N/A"
min_stp_str = f"{min_stp:.1f}" if not np.isnan(min_stp) else "N/A"
avg_ws_str = f"{avg_ws:.1f}" if not np.isnan(avg_ws) else "N/A"
max_ws_str = f"{max_ws:.1f}" if not np.isnan(max_ws) else "N/A"

text_stats = (f"Duration: {duration_hr}h  |  "
              f"Avg WS: {avg_ws_str} m/s  |  Max WS: {max_ws_str} m/s  |  "
              f"Min SLP: {min_slp_str} hPa  |  Min StP: {min_stp_str} hPa")

plt.figtext(0.5, 0.02, text_stats, ha='center', fontsize=11, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

plt.tight_layout()
# 预留底部空间给统计文本
plt.subplots_adjust(bottom=0.15)

plt.show()

# 如果需要保存图片，请取消下面注释
# fig.savefig(f'Typhoon_{target_typhoon_id}_Station_{target_station_id}.png', dpi=150, bbox_inches='tight')