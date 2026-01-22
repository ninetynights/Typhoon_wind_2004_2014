import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 绘图设置 (Mac 深度优化) =================
# 字体优先级：苹方 > 冬青 > 黑体 > Arial Unicode (防止中文乱码)
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False 
sns.set_style("whitegrid", {"font.sans-serif": ['PingFang SC', 'Heiti TC']})

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"
DATASET_PATH = os.path.join(BASE_DIR, "输出_机器学习", "Final_Training_Dataset_XGBoost.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "输出_机器学习", "模型与SHAP分析结果_v4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- [新增] 特征中英文映射字典 (用于汇报图表) ---
FEATURE_MAP_CN = {
    'Ty_Pressure': '台风中心气压',
    'Ty_Center_Wind': '台风近中心风速',
    'Ty_Lat': '台风纬度',
    'Ty_Lon': '台风经度',
    'Sta_Height': '站点海拔',
    'Dist_to_Coast': '离岸距离',
    'Terrain_15km': '地形开阔度(15km)',
    'Dist_Station_Ty': '站点与台风距离',
    'Azimuth_Station_Ty': '站点相对方位角'
}

# ================= 1. 数据准备 =================

def load_and_prep_data(filepath):
    print(f">>> 正在读取数据集: {filepath}")
    df = pd.read_csv(filepath)
    
    # 特征定义 (英文，用于模型训练)
    feature_cols = [
        # 动力因子
        'Ty_Pressure', 'Ty_Center_Wind', 'Ty_Lat', 'Ty_Lon',
        # 环境因子
        'Sta_Height', 'Dist_to_Coast', 'Terrain_15km',
        # 几何因子
        'Dist_Station_Ty', 'Azimuth_Station_Ty'
    ]
    target_col = 'Obs_Wind_Speed'
    
    print(f"    数据集样本数: {len(df)}")
    
    # 简单清洗
    if df[feature_cols].isnull().any().any():
        df = df.dropna(subset=feature_cols + [target_col])
    
    return df, feature_cols, target_col

# ================= 2. 模型训练与评估 (新增 KDE & 数据保存) =================

def train_model(df, features, target):
    print("\n>>> 开始划分数据集...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df['TID']))
    
    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx][target]
    X_test = df.iloc[test_idx][features]
    y_test = df.iloc[test_idx][target]
    
    print(f"    训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    print("\n>>> 开始训练 XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # --- 评估指标 ---
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"    [评估结果] R2: {r2:.4f}, RMSE: {rmse:.4f} m/s, MAE: {mae:.4f} m/s")
    
    # --- [新增] 保存模型评估指标到 CSV ---
    metrics_df = pd.DataFrame({
        'Metric': ['R2', 'RMSE', 'MAE', 'Train_Samples', 'Test_Samples'],
        'Value': [r2, rmse, mae, len(X_train), len(X_test)]
    })
    metrics_csv_path = os.path.join(OUTPUT_DIR, "Model_Performance_Metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
    print(f"    [保存] 模型指标已保存至: {metrics_csv_path}")

    # --- [新增] 保存预测值与观测值对比数据 (用于后续画 KDE 或散点图) ---
    pred_obs_df = pd.DataFrame({
        'Observed_Wind': y_test.values,
        'Predicted_Wind': y_pred
    })
    pred_obs_csv_path = os.path.join(OUTPUT_DIR, "Model_Prediction_vs_Observation.csv")
    pred_obs_df.to_csv(pred_obs_csv_path, index=False, encoding='utf-8-sig')
    print(f"    [保存] 预测对比数据已保存至: {pred_obs_csv_path}")

    # --- 绘图 1: 散点图 (Scatter) ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.1, s=3, color='#1f77b4')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("观测风速 (m/s)")
    plt.ylabel("预测风速 (m/s)")
    plt.title(f"模型预测性能\nR2={r2:.3f}, RMSE={rmse:.3f}")
    plt.savefig(os.path.join(OUTPUT_DIR, "1_Model_Scatter_CN.png"), dpi=300)
    plt.close()
    
    # --- 绘图 2: KDE 分布对比图 (Kernel Density Estimation) ---
    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_test, fill=True, label='观测值 (Observed)', color='blue', alpha=0.3)
    sns.kdeplot(y_pred, fill=True, label='预测值 (Predicted)', color='orange', alpha=0.3)
    plt.xlabel("风速 (m/s)")
    plt.ylabel("概率密度 (Density)")
    plt.title("观测值与预测值分布对比 (KDE Plot)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "2_Model_KDE_Distribution_CN.png"), dpi=300)
    plt.close()
    
    return model

# ================= 3. SHAP 分析 (CSV导出 + 中文绘图) =================

def run_shap_analysis(model, X, output_subdir, title_suffix=""):
    save_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(save_path, exist_ok=True)
    
    # --- 1. 计算 SHAP 值 ---
    # 限制最大解释样本数为 5000 加速
    MAX_SHAP_SAMPLES = 5000
    if len(X) > MAX_SHAP_SAMPLES:
        X_sample = X.sample(MAX_SHAP_SAMPLES, random_state=42)
    else:
        X_sample = X
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # --- 2. 导出 SHAP 重要性数据到 CSV ---
    # 计算平均绝对 SHAP 值
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_sample.columns
    
    importance_df = pd.DataFrame({
        'Feature_En': feature_names,
        'Feature_Cn': [FEATURE_MAP_CN.get(f, f) for f in feature_names],
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    csv_name = f"SHAP_Importance_{title_suffix}.csv".replace(" ", "_").replace("(", "").replace(")", "")
    importance_df.to_csv(os.path.join(save_path, csv_name), index=False, encoding='utf-8-sig')
    
    # --- 3. 准备中文显示数据 ---
    # 复制一份 X_sample 并重命名列为中文，仅用于绘图显示，不影响数值
    X_display_cn = X_sample.rename(columns=FEATURE_MAP_CN)
    
    # --- 4. 绘图 (中文版) ---
    
    # (A) 蜂群图 (Beeswarm)
    plt.figure(figsize=(10, 8))
    # 注意：传入 shap_values 和 X_display_cn，SHAP 会自动使用 X_display_cn 的列名作为标签
    shap.summary_plot(shap_values, X_display_cn, show=False)
    plt.title(f"特征重要性蜂群图 {title_suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"SHAP_Beeswarm_CN_{title_suffix}.png"), dpi=300)
    plt.close()
    
    # (B) 柱状图 (Bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display_cn, plot_type="bar", show=False)
    plt.title(f"特征平均贡献度 {title_suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"SHAP_Bar_CN_{title_suffix}.png"), dpi=300)
    plt.close()
    
    # (C) 依赖图 (Dependence) - 挑选 Top 4 特征
    # 从 importance_df 中获取前4个最重要的中文特征名
    top_4_features_cn = importance_df['Feature_Cn'].head(4).tolist()
    # 对应的英文名用于索引 shap_values
    top_4_features_en = importance_df['Feature_En'].head(4).tolist()
    
    for feat_en, feat_cn in zip(top_4_features_en, top_4_features_cn):
        plt.figure(figsize=(8, 6))
        # 这里的 trick 是：用原始 shap_values，但传入中文的 X_display_cn
        # SHAP dependence_plot 会根据列名匹配
        try:
            shap.dependence_plot(
                feat_cn, 
                shap_values, 
                X_display_cn, 
                interaction_index='auto', 
                show=False,
                alpha=0.5
            )
            plt.title(f"{feat_cn} 对风速的影响分析 {title_suffix}", fontsize=12)
            plt.ylabel("SHAP值 (对风速的贡献 m/s)")
            plt.tight_layout()
            clean_name = f"SHAP_Dep_{feat_en}_{title_suffix}".replace(" ", "_")
            plt.savefig(os.path.join(save_path, f"{clean_name}_CN.png"), dpi=300)
        except Exception as e:
            print(f"  [绘图警告] 依赖图 {feat_cn} 绘制失败: {e}")
        finally:
            plt.close()

# ================= 主程序 =================

def main():
    print(">>> 1. 数据加载...")
    df, features, target = load_and_prep_data(DATASET_PATH)
    
    print("\n>>> 2. 模型训练与评估...")
    model = train_model(df, features, target)
    
    print("\n>>> 3. SHAP 归因分析 (生成 CSV 和 中文图表)...")
    
    # 关闭所有之前的图形，防止内存溢出
    plt.close('all')
    
    # 全局分析
    print("  -> 执行全局分析...")
    run_shap_analysis(model, df[features], "全局分析", "All")
    
    # 分聚类分析 (8-10级)
    print("\n  -> 执行 8-10级 分聚类分析...")
    clusters = sorted(df['Cluster_8_10'].dropna().unique())
    for cid in clusters:
        if cid == -1: continue
        subset = df[df['Cluster_8_10'] == cid]
        if len(subset) > 200:
            print(f"     类别 {int(cid)} (N={len(subset)})")
            run_shap_analysis(model, subset[features], "8-10级分型", f"C{int(cid)}")
            
    # 分聚类分析 (11级+)
    print("\n  -> 执行 11级+ 分聚类分析...")
    clusters_11 = sorted(df['Cluster_11_Plus'].dropna().unique())
    for cid in clusters_11:
        if cid == -1: continue
        subset = df[df['Cluster_11_Plus'] == cid]
        if len(subset) > 200:
            print(f"     类别 {int(cid)} (N={len(subset)})")
            run_shap_analysis(model, subset[features], "11级以上分型", f"C{int(cid)}")

    print(f"\n[完成] 成果已保存至: {OUTPUT_DIR}")
    print("包括: 散点图, KDE分布图, 中文SHAP图(蜂群/柱状/依赖), 以及 CSV数据表(Model_Metrics, Prediction_vs_Observation)。")

if __name__ == "__main__":
    main()