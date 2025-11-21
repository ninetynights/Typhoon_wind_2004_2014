"""
查看_站点检验.py — 单站点台风影响时序快速可视化

目的与功能（整体说明）：
- 本脚本用于快速检查并可视化指定台风在指定观测站的逐小时风速与风向（含风向杆/Barbs）。
- 通过读取 NetCDF 文件中的台风内部索引、风速（wind_velocity）、风向（wind_direction）和时间（INITTIME）变量，
  提取目标台风在目标站点的影响时间段并绘制时序图，方便人工质检与逐站排查异常。

主要处理步骤：
1. 解析 NetCDF 中的属性映射（如 id_to_index、index_to_cn、index_to_en）以获得台风内部索引与中英文名称；
2. 根据用户指定的 target_typhoon_id 与 target_station_id 查找对应内部索引与站点索引；
3. 从 NetCDF 中提取该站点在该台风影响期的风速、风向与对应时间点；
4. 计算风向分量（U/V，用于绘制风向杆）并统计简单指标（平均风速、最大风速、持续时长）；
5. 绘制可视化图表：风速折线、风向杆、每小时风速标注，以及时间轴格式化与统计信息展示。

输入（需在脚本顶部设置或修改）：
- target_typhoon_id: 字符串，台风外部编号（与 NetCDF 属性 id_to_index 中键一致）
- target_station_id: 字符串，观测站编号（与 NetCDF 变量 STID 中值一致）
- nc_path: 指向包含变量 wind_velocity/wind_direction/typhoon_id_index/INITTIME/STID 的 NetCDF 文件路径

输出与展示：
- 在屏幕上弹出一幅图（matplotlib），显示该台风在该站点的逐小时最大风速与风向杆；
- 图中包含：每小时风速折线、每点风速文本标注、风向杆、统计行（持续时长、平均/最大风速）；
- 脚本不默认写出文件；如需保存图片，可在 plt.show() 前调用 fig.savefig(...)。
"""

from netCDF4 import Dataset, num2date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import re

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# 用户输入设置
target_typhoon_id = '2421'
target_station_id = '58446'
nc_path = '/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据/All_Typhoons_ExMaxWind_2004_2024.nc'

# 打开 NetCDF 文件
nc = Dataset(nc_path)

# 解析映射属性
def parse_mapping(attr_str):
    pairs = attr_str.strip().split(";")
    return {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in pairs if ":" in p)}

id_to_index = parse_mapping(nc.getncattr('id_to_index'))
index_to_cn = parse_mapping(nc.getncattr('index_to_cn'))
index_to_en = parse_mapping(nc.getncattr('index_to_en'))

# 获取台风索引及名称
if target_typhoon_id not in id_to_index:
    raise ValueError(f"台风编号 {target_typhoon_id} 不在数据中")
typhoon_index = int(id_to_index[target_typhoon_id])

# 获取台风中英文名
cn_name = index_to_cn[str(typhoon_index)]
en_name = index_to_en[str(typhoon_index)]

# 查找站点索引
stid_list = nc.variables['STID'][:].astype(str)
if target_station_id not in stid_list:
    raise ValueError(f"站点编号 {target_station_id} 不在数据中")
station_index = np.where(stid_list == target_station_id)[0][0]

# 提取数据
typhoon_ids = nc.variables['typhoon_id_index'][:, 0, station_index]
wind_speeds = nc.variables['wind_velocity'][:, 0, station_index]
wind_dirs = nc.variables['wind_direction'][:, 0, station_index]
init_times = nc.variables['INITTIME'][:]
init_units = nc.variables['INITTIME'].units
time_dt = num2date(init_times, init_units)

# 找出台风影响时间段索引
valid_idx = np.where(typhoon_ids == typhoon_index)[0]
if len(valid_idx) == 0:
    raise ValueError("该站点没有此台风影响数据")

# 数据处理
times = [datetime.datetime(t.year, t.month, t.day, t.hour, t.minute) for t in time_dt[valid_idx]]
ws = wind_speeds[valid_idx]
wd = wind_dirs[valid_idx]

# 计算风速方向分量（单位向量 * 风速）
theta_rad = np.deg2rad(wd)
U = -ws * np.sin(theta_rad)
V = -ws * np.cos(theta_rad)

# 计算平均风速与总时长
avg_ws = np.nanmean(ws)
max_ws = np.nanmax(ws)
duration_hr = len(times)

# 可视化
fig, ax = plt.subplots(figsize=(14, 6))

# 折线图：风速
ax.plot(times, ws, color='darkgrey', label='Max Wind Speed (m/s)', linewidth=0.8, zorder=3)

# 标注每个点的风速数值（位于点下方）
for x, y in zip(times, ws):
    if not np.isnan(y):
        ax.text(x, y - 0.1, f"{y:.1f}", ha='center', va='top', fontsize=8, color='grey', zorder=5)

# 绘制风向杆（barbs）
ax.barbs(times, ws, U, V, length=6, color='black', linewidth=1,
         barb_increments=dict(half=2, full=4, flag=20), zorder=4)

# 格式化时间轴
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
fig.autofmt_xdate(rotation=45)

# 标签和标题
ax.set_xlabel('Time (Hourly)')
ax.set_ylabel('Wind Speed (m/s)')
ax.set_title(f"Typhoon {target_typhoon_id} - {cn_name} ({en_name}) - Station {target_station_id}\n"
             f"Hourly Max Wind Speed and Direction", fontsize=14)

# 在图下方显示统计信息
text_stats = f"Duration: {duration_hr} hours  |  Avg Wind Speed: {avg_ws:.2f} m/s  |  Max Wind Speed: {max_ws:.2f} m/s"
plt.figtext(0.5, 0.01, text_stats, ha='center', fontsize=11)

# 其他设置
ax.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=0)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()