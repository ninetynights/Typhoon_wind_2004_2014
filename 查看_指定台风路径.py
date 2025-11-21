"""
查看_指定台风路径.py — 台风路径可视化工具（2010–2024 年，影响浙江）

总体目的：
- 从最佳路径文本数据与 NetCDF 元数据中筛选出影响浙江的台风，
  并在地图上可视化其路径、起终点时间及强度等级（使用不同颜色/标记表示等级）。

主要功能：
- 解析 NetCDF 文件属性（id_to_index）以获取在数据集中影响浙江的台风编号集合；
- 读取多年度的最佳路径 TXT（CHYYYYBST.txt）文件，抽取属于上述编号的路径点（时间、经纬度、强度）；
- 支持按用户输入绘制单个、多个或全部台风路径，路径按台风编号着色并标注起讫时间；
- 在路径上按时刻绘制强度点（不同符号/颜色表示起点、中间点、终点与强度等级），并绘制图例、网格与省界；
- 可选择将图像保存为文件或直接弹窗显示。

输入：
- folder: 包含 CHYYYYBST.txt 多年台风最佳路径的目录（脚本中变量 folder）
- nc_file: 包含属性 id_to_index 的 NetCDF 文件（脚本中变量 nc_file）
- 用户通过交互输入指定要绘制的台风编号（支持逗号或全角逗号分隔、输入 'all' 绘制全部）

输出：
- 在屏幕上显示绘制的地图（默认），或当传入 output_path 时将图片保存为 PNG（高分辨率，bbox_inches='tight'）
- 在控制台打印数据载入、可用台风列表与无效输入提示

"""

import os
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 台风等级颜色映射
LEVEL_COLORS = {
    "TD": "#A0A0A0",
    "TS": "#00CED1",
    "STS": "#1E90FF",
    "TY": "#FF8C00",
    "STY": "#FF4500",
    "SuperTY": "#8B0000",
    "Unknown": "black",
    "ET": "green"  # Extra-tropical
}

# 强度编号与等级映射（来自国家标准）
LEVEL_MAP = {
    "0": "Unknown",  # 弱于热带低压或未知
    "1": "TD",
    "2": "TS",
    "3": "STS",
    "4": "TY",
    "5": "STY",
    "6": "SuperTY",
    "9": "ET"  # 变性
}


def get_typhoon_ids_from_nc(nc_path):
    """从NC文件中读取所有影响浙江的台风编号"""
    nc = Dataset(nc_path)
    id_attr = nc.getncattr('id_to_index')
    id_map = {k.strip(): int(v.strip()) for k, v in [p.split(":") for p in id_attr.split(";") if ":" in p]}
    nc.close()
    return set(id_map.keys())


def read_selected_typhoon_tracks(folder_path, valid_ids):
    """读取所有txt文件中属于valid_ids的路径数据"""
    track_data = {}  # key: 中国台风编号，value: list of (time, lat, lon, level)

    for year in range(2010, 2025):
        filename = f"CH{year}BST.txt"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            current_id = None
            current_path = []

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("66666"):
                    if current_id and current_id in valid_ids and current_path:
                        track_data[current_id] = current_path
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 5:
                        current_id = parts[4]  # 中国台风编号 EEEE
                        current_path = []
                    else:
                        current_id = None
                        current_path = []
                elif current_id:
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 6:
                        try:
                            time_str = parts[0]
                            lat = float(parts[2]) / 10.0
                            lon = float(parts[3]) / 10.0
                            level_code = parts[1]  # I 位，强度标记
                            level = LEVEL_MAP.get(level_code, "Unknown")

                            current_path.append((time_str, lat, lon, level))
                        except:
                            continue

            if current_id and current_id in valid_ids and current_path:
                track_data[current_id] = current_path

    return track_data


def plot_typhoon_tracks(track_data, specific_ids=None, output_path=None,
                        title="2010–2024年影响浙江的台风路径"):
    """
    绘制台风路径
    :param track_data: 所有台风路径数据
    :param specific_ids: 需要绘制的特定台风ID列表（例如['2421', '2425']），若为None则绘制所有
    :param output_path: 图片输出路径
    :param title: 图片标题
    """
    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置地图范围（中国东部及周边区域）
    ax.set_extent([105, 140, 10, 45], crs=ccrs.PlateCarree())

    # 添加地理特征
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f5f5f5")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e0f7fa")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"))
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#e0f7fa")

    # 添加网格线和省份边界
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 添加省份
    provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linewidth=0.5)

    # 确定要绘制的台风ID
    if specific_ids is None:
        ids_to_plot = list(track_data.keys())
        plot_title = title
    else:
        # 处理全角和半角逗号
        processed_ids = []
        for tid in specific_ids:
            # 分割可能包含全角逗号的字符串
            if '，' in tid:
                processed_ids.extend(tid.split('，'))
            elif ',' in tid:
                processed_ids.extend(tid.split(','))
            else:
                processed_ids.append(tid)

        # 去除空白字符
        ids_to_plot = [tid.strip() for tid in processed_ids if tid.strip()]

        # 检查有效性
        valid_ids = [tid for tid in ids_to_plot if tid in track_data]
        invalid_ids = set(ids_to_plot) - set(valid_ids)

        if invalid_ids:
            print(f"警告: 以下台风ID不在数据集中: {', '.join(invalid_ids)}")

        if valid_ids:
            ids_to_plot = valid_ids
            plot_title = f"台风路径 ({'、'.join(valid_ids)})"
        else:
            print("错误: 没有找到有效的台风ID!")
            plt.close(fig)
            return

    # 颜色循环，用于区分多个台风
    COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    # 绘制路径
    for i, tid in enumerate(ids_to_plot):
        path = track_data[tid]
        if len(path) < 2:
            continue

        lons = [p[2] for p in path]
        lats = [p[1] for p in path]
        times = [p[0] for p in path]
        levels = [p[3] for p in path]

        # 为每个台风选择颜色
        color = COLORS[i % len(COLORS)]

        # 绘制路径线
        ax.plot(lons, lats, color=color, linewidth=2.5, alpha=0.9,
                transform=ccrs.PlateCarree(), zorder=3, label=tid)

        # 绘制强度点
        for j, (lon, lat, lvl) in enumerate(zip(lons, lats, levels)):
            marker_size = 50 if j == 0 else 40 if j == len(lons) - 1 else 30
            marker = 'o' if j == 0 else 's' if j == len(lons) - 1 else '.'
            ax.scatter(lon, lat, color=LEVEL_COLORS.get(lvl, "black"),
                       s=marker_size, marker=marker, edgecolor='white', linewidth=0.5,
                       transform=ccrs.PlateCarree(), zorder=4)

        # 标注起点和终点时间
        ax.text(lons[0], lats[0], f"{times[0][4:6]}/{times[0][6:8]}",
                fontsize=8, ha='right', va='bottom', transform=ccrs.PlateCarree())
        ax.text(lons[-1], lats[-1], f"{times[-1][4:6]}/{times[-1][6:8]}",
                fontsize=8, ha='left', va='top', transform=ccrs.PlateCarree())

    # 添加图例
    ax.legend(title="台风编号", loc="lower left", fontsize=9)

    # 添加强度图例
    handles = []
    for lvl, color in LEVEL_COLORS.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=8, label=lvl))
    ax.legend(handles=handles, title="强度等级", loc="lower right", fontsize=8)

    # 添加标题
    plt.title(plot_title, fontsize=16, pad=20)

    # 添加比例尺和指北针
    ax.add_artist(plt.Arrow(0.05, 0.05, 0.02, 0, transform=ax.transAxes, color='k'))
    ax.text(0.05, 0.05, 'N', transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='center')
    ax.text(0.13, 0.05, '1000 km', transform=ax.transAxes, fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {output_path}")
    else:
        plt.show()


def list_available_typhoons(track_data):
    """列出所有可用的台风ID"""
    print("\n可用的台风编号列表:")
    print("-" * 40)
    for i, tid in enumerate(sorted(track_data.keys())):
        print(f"{tid}", end="\t")
        if (i + 1) % 8 == 0:
            print()  # 每8个换行
    print("\n" + "-" * 40)


if __name__ == "__main__":
    folder = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
    nc_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"

    # 获取所有影响浙江的台风编号
    print("正在从NC文件中读取台风编号...")
    valid_typhoon_ids = get_typhoon_ids_from_nc(nc_file)

    # 读取所有台风路径数据
    print("正在读取台风路径数据...")
    track_data = read_selected_typhoon_tracks(folder, valid_typhoon_ids)
    print(f"成功加载 {len(track_data)} 个台风的路径数据")

    # 列出可用台风
    list_available_typhoons(track_data)

    # 用户交互
    while True:
        user_input = input(
            "\n请输入要绘制的台风编号（多个编号用逗号分隔，输入 'all' 绘制所有，输入 'exit' 退出）: ").strip()

        if user_input.lower() == 'exit':
            print("程序已退出")
            break

        if user_input.lower() == 'all':
            specific_ids = None
            title = "2010–2024年影响浙江的台风路径"
        elif user_input:
            # 直接使用输入的字符串，会在绘图函数中处理
            specific_ids = [user_input]
            title = ""  # 标题将在绘图函数中生成
        else:
            print("输入无效，请重新输入")
            continue

        # 绘制选定台风
        plot_typhoon_tracks(
            track_data,
            specific_ids=specific_ids,
            title=title
        )

        continue_input = input("是否继续绘制其他台风？(y/n): ").strip().lower()
        if continue_input != 'y':
            print("程序已退出")
            break