from netCDF4 import Dataset

# 你的文件路径
nc_path = '/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/数据/All_Typhoons_ExMaxWind_2004_2024.nc'

nc = Dataset(nc_path)

# 获取所有台风编号索引（维度：[typhoon, time, station]）
typhoon_ids = nc.variables['typhoon_id_index'][:, 0, :]
all_ty_ids = set()

# 遍历所有站点，收集非缺测编号
for i in range(typhoon_ids.shape[1]):
    ids = typhoon_ids[:, i]
    all_ty_ids.update(set(ids[ids >= 0]))

# 解析属性字符串为映射字典
def parse_mapping(attr_str):
    pairs = attr_str.strip().split(";")
    return {int(v.strip()): k.strip() for k, v in (p.split(":") for p in pairs if ":" in p)}

index_to_id = parse_mapping(nc.getncattr("id_to_index"))

# 获取有效台风编号
valid_ty_ids = [index_to_id[i] for i in sorted(all_ty_ids) if i in index_to_id]

# 打印台风编号总数与列表
print(f"台风总数：{len(valid_ty_ids)}")
print("台风编号：", valid_ty_ids)