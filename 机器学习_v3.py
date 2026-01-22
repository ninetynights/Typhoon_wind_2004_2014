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
OUTPUT_DIR = os.path.join(BASE_DIR, "输出_机器学习", "模型与SHAP分析结果_KDE")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 特征中英文映射字典 ---
FEATURE_MAP_CN = {
    'Ty_Pressure': '台风中心气压',
    'Ty_Center_Wind': '台风近中心风速',
    'Ty_Lat': '台风纬度',
    'Ty_Lon': '台风经度',
    'Sta_Height': '站点海拔',
    'Dist_to_Coast': '离岸距离',
    'Terrain_10km': '地形开阔度(10km)',
    'Dist_Station_Ty': '站点与台风距离',
    'Azimuth_Station_Ty': '站点相对方位角'
}

# ================= 1. 数据准备 =================

def load_and_prep_data(filepath):
    print(f">>> 正在读取数据集: {filepath}")
    df = pd.read_csv(filepath)
    
    # 特征定义
    feature_cols = [
        'Ty_Pressure', 'Ty_Center_Wind', 'Ty_Lat', 'Ty_Lon',
        'Sta_Height', 'Dist_to_Coast', 'Terrain_10km',
        'Dist_Station_Ty', 'Azimuth_Station_Ty'
    ]
    target_col = 'Obs_Wind_Speed'
    
    print(f"    数据集样本数: {len(df)}")
    
    if df[feature_cols].isnull().any().any():
        df = df.dropna(subset=feature_cols + [target_col])
    
    return df, feature_cols, target_col

# ================= 2. 模型训练与评估 =================

def train_model(df, features, target):
    print("\n>>> 开始划分数据集...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df['TID']))
    
    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx][target]
    X_test = df.iloc[test_idx][features]
    y_test = df.iloc[test_idx][target]
    
    print(f"    训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    print("\n>>> 开始训练 XGBoost (利用 M2 芯片)...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
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
    
    # 保存指标
    metrics_df = pd.DataFrame({
        'Metric': ['R2', 'RMSE', 'MAE', 'Train_Samples', 'Test_Samples'],
        'Value': [r2, rmse, mae, len(X_train), len(X_test)]
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "Model_Performance_Metrics.csv"), index=False, encoding='utf-8-sig')

    # 保存预测数据
    pred_obs_df = pd.DataFrame({'Observed_Wind': y_test.values, 'Predicted_Wind': y_pred})
    pred_obs_df.to_csv(os.path.join(OUTPUT_DIR, "Model_Prediction_vs_Observation.csv"), index=False, encoding='utf-8-sig')

    # 绘图: 散点
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.1, s=3, color='#1f77b4')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("观测风速 (m/s)")
    plt.ylabel("预测风速 (m/s)")
    plt.title(f"模型预测性能\nR2={r2:.3f}, RMSE={rmse:.3f}")
    plt.savefig(os.path.join(OUTPUT_DIR, "1_Model_Scatter_CN.png"), dpi=300)
    plt.close()
    
    # 绘图: KDE 对比
    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_test, fill=True, label='观测值 (Observed)', color='blue', alpha=0.3)
    sns.kdeplot(y_pred, fill=True, label='预测值 (Predicted)', color='orange', alpha=0.3)
    plt.xlabel("风速 (m/s)")
    plt.ylabel("概率密度 (Density)")
    plt.title("观测值与预测值分布对比")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "2_Model_KDE_Distribution_CN.png"), dpi=300)
    plt.close()
    
    return model

# ================= 3. SHAP 分析 (核心修改：添加边际分布 KDE) =================

def run_shap_analysis(model, X, output_subdir, title_suffix=""):
    save_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(save_path, exist_ok=True)
    
    # --- 1. 计算 SHAP 值 ---
    MAX_SHAP_SAMPLES = 5000
    if len(X) > MAX_SHAP_SAMPLES:
        X_sample = X.sample(MAX_SHAP_SAMPLES, random_state=42)
    else:
        X_sample = X
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # --- 2. 导出 SHAP 重要性 CSV ---
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
    X_display_cn = X_sample.rename(columns=FEATURE_MAP_CN)
    
    # --- 4. 绘图 (中文版) ---
    
    # (A) 蜂群图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display_cn, show=False)
    plt.title(f"特征重要性蜂群图 {title_suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"SHAP_Beeswarm_CN_{title_suffix}.png"), dpi=300)
    plt.close()
    
    # (B) 柱状图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display_cn, plot_type="bar", show=False)
    plt.title(f"特征平均贡献度 {title_suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"SHAP_Bar_CN_{title_suffix}.png"), dpi=300)
    plt.close()
    
    # (C) [核心修改] 依赖图 + 顶部 KDE 分布
    # 挑选 Top 4 特征
    top_4_features_cn = importance_df['Feature_Cn'].head(4).tolist()
    top_4_features_en = importance_df['Feature_En'].head(4).tolist()
    
    for feat_en, feat_cn in zip(top_4_features_en, top_4_features_cn):
        try:
            # 创建画布：使用 GridSpec 分割为 上(KDE) 下(SHAP) 两部分
            # height_ratios=[1.2, 5] 表示上方高度是下方的约 1/4
            fig = plt.figure(figsize=(9, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 5], hspace=0.08)
            
            ax_kde = fig.add_subplot(gs[0])
            ax_shap = fig.add_subplot(gs[1])
            
            # 1. 绘制上方 KDE 图 (显示该特征的数据分布密度)
            # 注意：x轴共享
            sns.kdeplot(
                x=X_display_cn[feat_cn], 
                ax=ax_kde, 
                fill=True, 
                color='#4c72b0', 
                alpha=0.3, 
                linewidth=1.5
            )
            ax_kde.set_xlabel("") # 移除X标签
            ax_kde.set_xticklabels([]) # 移除X刻度
            ax_kde.set_yticks([]) # 移除Y刻度(密度值对非统计专业不直观，看形状即可)
            ax_kde.set_ylabel("")
            ax_kde.set_title(f"{feat_cn} 分布与 SHAP 贡献分析 {title_suffix}", fontsize=13, pad=10)
            
            # 去除边框，更美观
            sns.despine(ax=ax_kde, left=True, bottom=True)
            
            # 2. 绘制下方 SHAP 依赖图
            # ax=ax_shap 将图画在指定的子图上
            shap.dependence_plot(
                feat_cn, 
                shap_values, 
                X_display_cn, 
                interaction_index='auto', 
                show=False,
                ax=ax_shap,
                alpha=0.6,
                x_jitter=0 # 如果数据点太密集，可以设为非0
            )
            
            # 强制对齐 X 轴范围 (这一步很重要)
            # 获取 SHAP 图自动设定的 X 轴范围
            shap_xlim = ax_shap.get_xlim()
            ax_kde.set_xlim(shap_xlim)
            
            # 调整标签
            ax_shap.set_ylabel("SHAP值 (对风速的贡献 m/s)")
            
            # 保存
            clean_name = f"SHAP_Dep_KDE_{feat_en}_{title_suffix}".replace(" ", "_")
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
    
    print("\n>>> 3. SHAP 归因分析 (带 KDE 分布图)...")
    
    plt.close('all')
    
    print("  -> 执行全局分析...")
    run_shap_analysis(model, df[features], "全局分析", "All")
    
    print("\n  -> 执行 8-10级 分聚类分析...")
    clusters = sorted(df['Cluster_8_10'].dropna().unique())
    for cid in clusters:
        if cid == -1: continue
        subset = df[df['Cluster_8_10'] == cid]
        if len(subset) > 200:
            print(f"     类别 {int(cid)} (N={len(subset)})")
            run_shap_analysis(model, subset[features], "8-10级分型", f"C{int(cid)}")
            
    print("\n  -> 执行 11级+ 分聚类分析...")
    clusters_11 = sorted(df['Cluster_11_Plus'].dropna().unique())
    for cid in clusters_11:
        if cid == -1: continue
        subset = df[df['Cluster_11_Plus'] == cid]
        if len(subset) > 200:
            print(f"     类别 {int(cid)} (N={len(subset)})")
            run_shap_analysis(model, subset[features], "11级以上分型", f"C{int(cid)}")

    print(f"\n[完成] 成果已保存至: {OUTPUT_DIR}")
    print("新增特性: 所有单变量依赖图 (SHAP Dependence Plot) 顶部均附带 KDE 数据分布图。")

if __name__ == "__main__":
    main()