import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 绘图设置 ---
# 设置字体以支持中文显示
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['PingFang SC','Heiti TC'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"
DATASET_PATH = os.path.join(BASE_DIR, "输出_机器学习", "Final_Training_Dataset_XGBoost.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "输出_机器学习", "模型与SHAP分析结果")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 1. 数据准备 =================

def load_and_prep_data(filepath):
    print(f">>> 正在读取数据集: {filepath}")
    df = pd.read_csv(filepath)
    
    # 定义特征 (X) 和 目标 (y)
    # 去掉 ID 类信息，保留物理量
    feature_cols = [
        # --- 动力因子 (台风) ---
        'Ty_Pressure',       # 核心强度指标
        'Ty_Center_Wind',    # 辅助强度指标
        'Ty_Lat', 'Ty_Lon',  # 台风位置 (隐含了移动方向/路径特征)
        
        # --- 环境/下垫面因子 (站点) ---
        'Sta_Height',        # 海拔
        'Dist_to_Coast',     # 离岸距离 (海陆差异)
        'Terrain_10km',      # 地形复杂度 (代表地形粗糙度，选10km尺度较适中)
        # 'Terrain_5km', 'Terrain_15km', # 避免多重共线性，选一个代表即可，或者全放进去让树模型自己选
        
        # --- 几何因子 (相对关系) ---
        'Dist_Station_Ty',   # 距离衰减效应
        'Azimuth_Station_Ty' # 象限/方位效应 (危险半圆)
    ]
    
    target_col = 'Obs_Wind_Speed'
    
    print(f"    数据集样本数: {len(df)}")
    print(f"    使用特征 ({len(feature_cols)}个): {feature_cols}")
    
    # 检查缺失值
    if df[feature_cols].isnull().any().any():
        print("    [警告] 发现缺失值，正在填充...")
        df = df.dropna(subset=feature_cols + [target_col])
        print(f"    填充后样本数: {len(df)}")
        
    return df, feature_cols, target_col

# ================= 2. 模型训练 =================

def train_model(df, features, target):
    print("\n>>> 开始划分数据集 (按台风TID分组划分)...")
    
    # 关键步骤：按 TID 分组切分，防止同一场台风的数据同时出现在训练集和测试集 (Data Leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df['TID']))
    
    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx][target]
    X_test = df.iloc[test_idx][features]
    y_test = df.iloc[test_idx][target]
    
    print(f"    训练集: {len(X_train)} 样本 ({df.iloc[train_idx]['TID'].nunique()} 个台风)")
    print(f"    测试集: {len(X_test)} 样本 ({df.iloc[test_idx]['TID'].nunique()} 个台风)")
    
    print("\n>>> 开始训练 XGBoost 模型...")
    # 参数可以根据需要微调，这里使用一套比较稳健的回归参数
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # 评估
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n>>> 模型评估 (测试集):")
    print(f"    RMSE: {rmse:.4f} m/s")
    print(f"    MAE : {mae:.4f} m/s")
    print(f"    R2  : {r2:.4f}")
    
    # 绘制 真实值 vs 预测值 散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.1, s=3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("观测风速 (Observed Wind Speed)")
    plt.ylabel("预测风速 (Predicted Wind Speed)")
    plt.title(f"XGBoost 模型性能\nR2={r2:.3f}, RMSE={rmse:.3f}")
    plt.savefig(os.path.join(OUTPUT_DIR, "1_Model_Performance_Scatter.png"), dpi=300)
    plt.close()
    
    return model, X_train, X_test

# ================= 3. SHAP 分析 (核心) =================

def run_shap_analysis(model, X, df_meta, output_subdir, title_suffix=""):
    """
    运行 SHAP 分析并画图
    model: 训练好的模型
    X: 特征矩阵 (DataFrame)
    df_meta: 包含元数据(TID, Cluster等)的原始DataFrame，用于筛选
    output_subdir: 图片保存子目录
    """
    save_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n>>> 计算 SHAP 值 ({title_suffix})... (这可能需要一点时间)")
    
    # 使用 TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # 为了速度，如果是全量数据，可以下采样。但为了准确性，建议使用全量或较大样本
    # 这里我们使用传入的 X (通常建议用 X_train 的一部分作为背景，或者直接解释 X)
    # 如果样本量太大 (>10w)，可以采样前 10000 个样本进行解释
    if len(X) > 10000:
        X_sample = X.sample(10000, random_state=42)
    else:
        X_sample = X
        
    shap_values = explainer.shap_values(X_sample)
    
    # 1. Summary Plot (蜂群图) - 最经典的图
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP 特征重要性蜂群图 {title_suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"SHAP_Summary_Beeswarm_{title_suffix}.png"), dpi=300)
    plt.close()
    
    # 2. Bar Plot (平均绝对重要性)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP 特征平均重要性排名 {title_suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"SHAP_Importance_Bar_{title_suffix}.png"), dpi=300)
    plt.close()
    
    # 3. Dependence Plots (依赖图) - 针对物理意义强的因子
    # 重点关注：距离、方位角、地形
    target_features = ['Dist_Station_Ty', 'Azimuth_Station_Ty', 'Dist_to_Coast', 'Terrain_10km', 'Ty_Pressure']
    
    for feat in target_features:
        if feat in X.columns:
            plt.figure(figsize=(8, 6))
            # interaction_index 可以选自动，或者指定 'Ty_Pressure' 来看强度对该因子的调制
            shap.dependence_plot(feat, shap_values, X_sample, interaction_index='auto', show=False)
            plt.title(f"SHAP 依赖图: {feat} {title_suffix}")
            plt.tight_layout()
            # 文件名处理特殊字符
            fname = f"SHAP_Dependence_{feat}_{title_suffix}".replace(" ", "_")
            plt.savefig(os.path.join(save_path, f"{fname}.png"), dpi=300)
            plt.close()

# ================= 主程序 =================

def main():
    # 1. 加载数据
    df, features, target = load_and_prep_data(DATASET_PATH)
    
    # 2. 训练全局模型
    # 我们用所有数据训练一个能够理解普遍物理规律的“大脑”
    model, X_train, X_test = train_model(df, features, target)
    
    # 保存模型
    model.save_model(os.path.join(OUTPUT_DIR, "XGBoost_Typhoon_Wind.json"))
    
    # 3. 全局 SHAP 分析 (解释整体规律)
    run_shap_analysis(model, df[features], df, "全局分析", title_suffix="(All Data)")
    
    # ================= 4. 分聚类深入探究 (你的核心需求) =================
    print("\n" + "="*50)
    print(">>> 开始分聚类 SHAP 归因分析...")
    print("="*50)
    
    # 策略：不重新训练模型，而是用全局模型去解释特定聚类的数据。
    # 这样可以看出：对于特定的路径类型，模型是“看到”了什么特征才预测出大风的。
    
    # --- 分析 8-10 级聚类 ---
    # 聚类ID通常是 0, 1, 2, 3, 4 (共5类), -1是噪声
    unique_clusters_8_10 = sorted(df['Cluster_8_10'].dropna().unique())
    
    for cid in unique_clusters_8_10:
        if cid == -1: continue # 跳过噪声
        
        # 筛选出属于该类别的样本
        subset_mask = df['Cluster_8_10'] == cid
        subset_df = df[subset_mask]
        
        if len(subset_df) < 100: continue # 样本太少不分析
        
        print(f"  -> 分析 [8-10级聚类 - 类别 {cid}] (样本数: {len(subset_df)})")
        run_shap_analysis(model, subset_df[features], subset_df, "8-10级聚类分型", title_suffix=f"[8-10级_Cluster{cid}]")

    # --- 分析 11级+ 聚类 ---
    unique_clusters_11 = sorted(df['Cluster_11_Plus'].dropna().unique())
    
    for cid in unique_clusters_11:
        if cid == -1: continue
        
        subset_mask = df['Cluster_11_Plus'] == cid
        subset_df = df[subset_mask]
        
        if len(subset_df) < 100: continue
        
        print(f"  -> 分析 [11级以上聚类 - 类别 {cid}] (样本数: {len(subset_df)})")
        run_shap_analysis(model, subset_df[features], subset_df, "11级以上聚类分型", title_suffix=f"[11级+_Cluster{cid}]")

    print(f"\n[完成] 所有分析图表已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()