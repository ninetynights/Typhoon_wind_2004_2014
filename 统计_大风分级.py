"""
大风分级统计.py — 台风影响下观测站超阈值/精确等级小时数统计与可视化（分级版 V6）

目的与功能概述：
- 【双重功能】
- 1. (Exceed) 批量统计风速 【大于等于】 给定阈值（如 8级及以上, 9级及以上...）的小时数。
- 2. (Exact)  批量统计风速 【精确等于】 某个等级（如 8级, 9级...）的小时数。
- 
- 为每个阈值等级（如 "8级及以上" 或 "8级"）生成独立的输出目录。
- 为每个台风生成：
    * CSV 报表：每个站点的经纬、高度与统计小时数（始终生成）。
    * 地图图像：在地图上以“彩色数字”形式标注各站点的统计小时数（仅在该台风有站点超阈值时生成）。
- 生成总体（All Typhoons）汇总 CSV 与地图。
- 【新增】生成两个主汇总文件：
    * AllTyphoons_Exceed.csv：所有“大于等于”任务的汇总。
    * AllTyphoons_Exact.csv：所有“区间等于”任务的汇总。
- 地图上增加绘制市界（SHP_CITY_PATH）。
- 额外输出：终端列出在影响期间没有任一站点超过阈值的台风（按级别分别汇总）。

输入：
- NC_PATH（脚本顶部配置）：包含变量 wind_velocity、typhoon_id_index、STID、lat、lon、height 的 NetCDF 文件。
- SHP_CITY_PATH（脚本顶部配置）：市界 Shapefile 路径。
- 可通过脚本顶部参数调整：
    * WIND_LEVELS_EXCEED: "大于等于" 的阈值列表
    * WIND_LEVELS_EXACT: "区间等于" 的阈值列表
    * EXTENT: 地图经纬范围
    * SHOW_ZERO: 是否在地图上显示超阈值小时数为 0 的站点
    * BASE_OUTPUT_DIR: 输出的根目录

输出（保存至 BASE_OUTPUT_DIR 的子目录，如 "输出_8级及以上" 或 "输出_8级"）：
- {输出_X级}/csv/...：单台风每站统计小时数
- {输出_X级}/csv/AllTyphoons_...csv：所有台风汇总小时数
- {输出_X级}/figs/...png：单台风站点彩色数字地图
- {输出_X级}/figs/AllTyphoons_...png：总体汇总地图
- AllTyphoons_Exceed.csv：【Exceed】主汇总文件
- AllTyphoons_Exact.csv：【Exact】主汇总文件
- 终端打印详细的处理进度与“无超阈值台风”汇总信息

"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# ======= 1. NetCDF 路径 =======
NC_PATH    = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/Refined_Combine_Stations_ExMaxWind_Fixed.nc"

# ======= 2. 基础输出目录 =======
BASE_OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计")

# ======= 3. 【修改】风速分级阈值 (m/s) 与名称 =======

# 任务 1: "大于等于" (Exceed)
WIND_LEVELS_EXCEED = [
    (17.2, "8级及以上"),
    (20.8, "9级及以上"),
    (24.5, "10级及以上"),
    (28.5, "11级及以上"),
    (32.7, "12级及以上"),
    (37.0, "13级及以上"),
    (41.5, "14级及以上"),
    (46.2, "15级及以上"),
    (51.0, "16级及以上"),
]

# 任务 2: "区间等于" (Exact)
WIND_LEVELS_EXACT = [
    (17.2, 20.7, "8级"),
    (20.8, 24.4, "9级"),
    (24.5, 28.4, "10级"),
    (28.5, 32.6, "11级"),
    (32.7, 36.9, "12级"),
    (37.0, 41.4, "13级"),
    (41.5, 46.1, "14级"),
    (46.2, 50.9, "15级"),
    (51.0, 56.0, "16级"),
]


# ======= 4. Shapefile 路径 =======
SHP_CITY_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/shapefile/市界/浙江市界.shp"

# ======= 5. 其他绘图参数 =======
EXTENT     = [118, 123, 27, 31.5]   # 设为 None 则自适应
TEXT_SIZE  = 8                    # 数字字号
CMAP_NAME  = "viridis"            # 颜色映射
DRAW_GRID  = True                 # 是否画经纬网
SHOW_ZERO  = False                # 小时数为 0 时是否也标注

# ----------------------------- 小工具 -----------------------------
def parse_mapping(attr_str: str):
    """把全局属性里的 'a:1; b:2' 解析为 dict[str, str]"""
    if not attr_str:
        return {}
    pairs = (p for p in attr_str.strip().split(";") if ":" in p)
    return {k.strip(): v.strip() for k, v in (q.split(":", 1) for q in pairs)}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\\s]+", "_", name).strip("_")

def draw_station_count_text_map(
    lons, lats, counts, stids, title, out_png,
    threshold_val, # 用于色标
    extent=None, text_size=8, cmap_name="viridis", show_zero=True,
    vmin=None, vmax=None
):
    """
    用“彩色数字”标注每个站点的超阈值小时数。(函数不变)
    """
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    # 底图要素
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

    # 加载并绘制市界
    try:
        city_shapes = list(shpreader.Reader(SHP_CITY_PATH).geometries())
        ax.add_geometries(
            city_shapes, ccrs.PlateCarree(),
            edgecolor='gray', facecolor='none',
            linewidth=0.5, linestyle='--'
        )
    except Exception as e:
        print(f"\n[WARN] 无法加载市界 Shapefile: {SHP_CITY_PATH}")
        print(f"[WARN] 错误: {e}")
        print("[WARN] 将继续绘图（不含市界）。\n")
        pass

    # 范围
    if extent and len(extent) == 4:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons)-pad_lon, np.nanmax(lons)+pad_lon,
                       np.nanmin(lats)-pad_lat, np.nanmax(lats)+pad_lat],
                      crs=ccrs.PlateCarree())

    # 颜色映射
    if vmin is None:
        vmin = np.nanmin(counts)
    if vmax is None:
        vmax = np.nanmax(counts)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # 逐站点标注彩色数字
    for x, y, val in zip(lons, lats, counts):
        if (not show_zero) and (val == 0):
            continue
        color = cmap(norm(val if np.isfinite(val) else 0.0))
        txt = ax.text(
            x, y, f"{int(val)}", fontsize=text_size,
            ha='center', va='center', color=color,
            transform=ccrs.PlateCarree()
        )
        txt.set_path_effects([
            pe.withStroke(linewidth=1.5, foreground="white")
        ])

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    # 【修改】色标标签更通用
    cbar.set_label(f"统计小时数 (m/s)")

    # 网格
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, crs=ccrs.PlateCarree())
    try:
        gl.right_labels = False
        gl.top_labels = False
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ----------------------------- 【新增】核心处理函数 -----------------------------
def process_wind_level_task(
    task_name,                          # "Exceed" or "Exact"
    col_tag,                            # "gt" or "eq"
    thresh_min, thresh_max,            # The thresholds
    level_name, level_short_name,      # Names for folders and titles
    base_output_dir,                   # Root output dir
    stids, lons, lats, heights,        # Station data
    wind_speeds, ty_ids,               # NC data
    items, index_to_cn, index_to_en    # Typhoon lookup data
):
    """
    执行单项风速分级统计任务（Exceed 或 Exact）。
    遍历所有台风，生成 CSV 和图像，并返回总体统计。
    """
    
    # 动态生成标题字符串
    if thresh_max == np.inf:
        title_range_str = f"{level_short_name} (≥ {thresh_min:.1f} m/s)"
        print_range_str = f"≥ {thresh_min:.1f} m/s"
    else:
        title_range_str = f"{level_short_name} ({thresh_min:.1f} - {thresh_max:.1f} m/s)"
        print_range_str = f"[{thresh_min:.1f}, {thresh_max:.1f}] m/s"

    print(f"\n{'='*70}")
    print(f"[START] 任务: {task_name} - {level_name} ({title_range_str})")
    print(f"{'='*70}")

    # 1. 输出目录 (按级别)
    current_output_dir = base_output_dir / f"输出_{level_name}"
    out_csv = current_output_dir / "csv"
    out_fig = current_output_dir / "figs"
    ensure_dir(out_csv); ensure_dir(out_fig)

    print(f"[INFO] {level_name} 的输出根目录: {current_output_dir.resolve()}")

    # 2. 统计 + 输出 (为每个级别重置)
    n_sta = stids.shape[0]
    total_counts = np.zeros(n_sta, dtype=int)
    zero_exceed_typhoons = []

    print(f"开始遍历 {len(items)} 个台风过程，统计 {level_name} ({print_range_str}) 的情况...")

    for tid_str, ty_idx in items:
        cn_name = index_to_cn.get(str(ty_idx), "")
        en_name = index_to_en.get(str(ty_idx), "")

        print(f"  -> 正在统计: {tid_str} {cn_name} ({en_name})")

        counts = np.zeros(n_sta, dtype=int)
        for i in range(n_sta):
            mask = (ty_ids[:, i] == ty_idx)
            if not np.any(mask):
                counts[i] = 0
                continue
            ws = wind_speeds[mask, i]
            
            # 【核心修改】使用范围进行统计
            mask_min = (ws >= thresh_min)
            mask_max = (ws <= thresh_max)
            counts[i] = int(np.sum(mask_min & mask_max))
        
        sum_counts = np.sum(counts)

        if sum_counts == 0:
            msg = f"    [INFO] 台风 {tid_str} ({cn_name}) 期间，无站点风速在 {print_range_str} (将跳过此台风的 {level_name} 地图绘制)"
            print(msg)
            zero_exceed_typhoons.append(f"  - {tid_str} ({cn_name} / {en_name})")
        
        # CSV（单台风）- 始终生成
        col_name_single = f"Hours_{col_tag}_{thresh_min:.1f}"
        df = pd.DataFrame({
            "STID": stids,
            "Lon": lons,
            "Lat": lats,
            "Height": heights,
            col_name_single: counts
        })
        # 文件名使用任务名、等级、台风名
        fname = f"{task_name}_{level_name}_{sanitize_filename(tid_str)}_{sanitize_filename(cn_name)}_{sanitize_filename(en_name)}.csv"
        df.to_csv(out_csv / fname, index=False, encoding="utf-8")

        # 图（单台风）- 仅在 sum_counts > 0 时绘制
        if sum_counts > 0:
            title = f"台风编号 {tid_str} - {cn_name} ({en_name})\n{title_range_str} 总小时数"
            png_name = f"{task_name}_{level_name}_{sanitize_filename(tid_str)}.png"
            png = out_fig / png_name
            draw_station_count_text_map(
                lons, lats, counts, stids, title, str(png),
                threshold_val=thresh_min,
                extent=EXTENT, text_size=TEXT_SIZE, cmap_name=CMAP_NAME, show_zero=SHOW_ZERO
            )

        total_counts += counts

    # 3. 总体 CSV + 图 (当前级别) - 始终生成
    print(f"  -> 正在生成 {level_name} 的总体汇总...")
    
    col_name_total = f"TotalHours_{col_tag}_{thresh_min:.1f}"
    df_total = pd.DataFrame({
        "STID": stids,
        "Lon": lons,
        "Lat": lats,
        "Height": heights,
        col_name_total: total_counts
    })
    total_csv_path = out_csv / f"AllTyphoons_{task_name}_{level_name}.csv"
    df_total.to_csv(total_csv_path, index=False, encoding="utf-8")
    
    # 绘制总图
    title_total = f"2010-2024大风影响台风过程\n小时极大风{title_range_str} 总小时数"
    total_png_path = out_fig / f"AllTyphoons_{task_name}_{level_name}.png"
    draw_station_count_text_map(
        lons, lats, total_counts, stids, title_total, str(total_png_path),
        threshold_val=thresh_min,
        extent=EXTENT, text_size=TEXT_SIZE, cmap_name=CMAP_NAME, show_zero=SHOW_ZERO
    )

    # 4. 打印当前级别的汇总
    print("\n" + "="*50)
    print(f"           *** {level_name} 统计完毕 ***")
    print(f"[OK] {level_name} 单台风 CSV/图 已输出至: {out_csv.resolve()} 和 {out_fig.resolve()}")
    print(f"[OK] {level_name} 总体 CSV 已保存: {total_csv_path.resolve()}")
    print(f"[OK] {level_name} 总体 图 已保存: {total_png_path.resolve()}")
    
    if zero_exceed_typhoons:
        print(f"\n[SUMMARY] ({level_name}) 以下 {len(zero_exceed_typhoons)} 个台风影响期间，无站点风速在 {print_range_str}：")
        for msg in zero_exceed_typhoons:
            print(msg)
    else:
        print(f"\n[SUMMARY] ({level_name}) 所有台风影响期间，均有站点风速在 {print_range_str}。")
    print("="*50 + "\n")

    # 5. 返回总统计结果
    return total_counts, col_name_total


# ----------------------------- 主逻辑 (已重构) -----------------------------
def main():
    # 1) 读数据 (只读一次)
    print(f"正在读取 NetCDF 文件: {NC_PATH}")
    nc = Dataset(NC_PATH)

    # 全局属性映射
    id_to_index = parse_mapping(getattr(nc, 'id_to_index', None))
    index_to_cn = parse_mapping(getattr(nc, 'index_to_cn', None))
    index_to_en = parse_mapping(getattr(nc, 'index_to_en', None))

    # 站点信息
    stids   = np.array(nc.variables['STID'][:]).astype(str)
    lats    = np.array(nc.variables['lat'][:], dtype=float)
    lons    = np.array(nc.variables['lon'][:], dtype=float)
    heights = np.array(nc.variables['height'][:], dtype=float) if 'height' in nc.variables else np.full_like(lats, np.nan)

    # 风速/台风索引
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    ty_ids      = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)

    # 将 ty_ids 统一为整数
    if np.issubdtype(ty_ids.dtype, np.floating):
        ty_ids_int = np.full_like(ty_ids, fill_value=-1, dtype=int)
        valid = ~np.isnan(ty_ids)
        ty_ids_int[valid] = ty_ids[valid].astype(int)
        ty_ids = ty_ids_int
    else:
        ty_ids = ty_ids.astype(int)

    n_time, n_sta = wind_speeds.shape

    # 3) 确定要遍历的台风索引与 TID 名称
    if id_to_index:
        items = []
        for tid_str, idx_str in id_to_index.items():
            try:
                idx = int(str(idx_str).strip())
            except Exception:
                continue
            items.append((tid_str, idx))
        items.sort(key=lambda kv: kv[0])
    else:
        uniq = sorted({int(x) for x in np.unique(ty_ids) if int(x) >= 0})
        items = [(str(idx), idx) for idx in uniq]
    
    print(f"数据读取完毕。共 {n_sta} 个站点，{len(items)} 个台风过程。")

    # 【修改】创建两个 Master DataFrame
    df_master_summary_exceed = pd.DataFrame({
        "STID": stids, "Lon": lons, "Lat": lats, "Height": heights
    })
    df_master_summary_exact = pd.DataFrame({
        "STID": stids, "Lon": lons, "Lat": lats, "Height": heights
    })

    # 准备共享参数
    common_args = {
        "base_output_dir": BASE_OUTPUT_DIR,
        "stids": stids, "lons": lons, "lats": lats, "heights": heights,
        "wind_speeds": wind_speeds, "ty_ids": ty_ids,
        "items": items, "index_to_cn": index_to_cn, "index_to_en": index_to_en
    }

    # -----------------------------------------------------
    # 任务 1: "大于等于" (Exceed)
    # -----------------------------------------------------
    print("\n" + "="*80)
    print(">>> 正在开始 任务 1: '大于等于' (Exceed) 批量统计 <<<")
    print("="*80 + "\n")
    
    for threshold, level_name in WIND_LEVELS_EXCEED:
        level_short_name = f"≥{level_name.split('级')[0]}级"
        
        total_counts, col_name_total = process_wind_level_task(
            task_name="Exceed", col_tag="gt",
            thresh_min=threshold, thresh_max=np.inf,
            level_name=level_name, level_short_name=level_short_name,
            **common_args
        )
        # 将总统计添加到 Master DataFrame
        df_master_summary_exceed[col_name_total] = total_counts

    # -----------------------------------------------------
    # 任务 2: "区间等于" (Exact)
    # -----------------------------------------------------
    print("\n" + "="*80)
    print(">>> 正在开始 任务 2: '区间等于' (Exact) 批量统计 <<<")
    print("="*80 + "\n")

    for thresh_min, thresh_max, level_name in WIND_LEVELS_EXACT:
        level_short_name = level_name # e.g., "8级"
        
        total_counts, col_name_total = process_wind_level_task(
            task_name="Exact", col_tag="eq",
            thresh_min=thresh_min, thresh_max=thresh_max,
            level_name=level_name, level_short_name=level_short_name,
            **common_args
        )
        # 将总统计添加到 Master DataFrame
        df_master_summary_exact[col_name_total] = total_counts


    # -----------------------------------------------------
    # 【修改】保存 两个 Master 汇总 CSV
    # -----------------------------------------------------
    print(f"\n[MASTER SUMMARY] 正在保存所有风速等级的总体汇总...")
    try:
        # 保存 Exceed 汇总
        master_summary_path_exceed = BASE_OUTPUT_DIR / "AllTyphoons_Exceed.csv"
        df_master_summary_exceed.to_csv(master_summary_path_exceed, index=False, encoding="utf-8")
        print(f"[OK] (Exceed) '大于等于' 汇总 CSV 已保存至: {master_summary_path_exceed.resolve()}")
        
        # 保存 Exact 汇总
        master_summary_path_exact = BASE_OUTPUT_DIR / "AllTyphoons_Exact.csv"
        df_master_summary_exact.to_csv(master_summary_path_exact, index=False, encoding="utf-8")
        print(f"[OK] (Exact) '区间等于' 汇总 CSV 已保存至: {master_summary_path_exact.resolve()}")

    except Exception as e:
        print(f"[ERROR] 保存 Master 汇总 CSV 失败: {e}")


    print("\n" + "="*80)
    print("          *** 所有分级统计任务全部完成 ***")
    print(f"所有结果均已保存至: {BASE_OUTPUT_DIR.resolve()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()