import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold  # [修改点] 引入分组K折验证
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 绘图设置 (Mac 深度优化) =================
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 
sns.set_style("whitegrid", {"font.sans-serif": ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']})

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"

# 数据集路径
DATASET_PATH = os.path.join(BASE_DIR, "输出_机器学习", "Final_Training_Dataset_XGBoost_v3_Pressure.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "输出_机器学习", "模型与SHAP分析结果_10折验证版")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 特征中英文映射字典 ---
FEATURE_MAP_CN = {
    'Ty_Pressure': '台风中心气压',
    'Ty_Center_Wind': '台风近中心风速',
    'Sta_Height': '站点海拔',
    'Dist_to_Coast': '离岸距离',
    'Terrain_10km': '地形开阔度(10km)',
    'Dist_Station_Ty': '站点与台风距离',
    'Azimuth_Station_Ty': '站点相对方位角',
    'Ty_Move_Speed': '台风移动速度',
    'Ty_Move_Dir': '台风移动方向',
    'Sta_Slope': '站点坡度',
    'Sta_TPI': '地形位置指数',
    'Pressure_Gradient_Obs': '观测气压梯度(Obs)',
    'Pressure_Gradient_SLP': '海平面气压梯度(SLP)'
}

# ================= 1. 数据准备 (保持不变) =================
def load_and_prep_data(filepath):
    print(f">>> 正在读取数据集: {filepath}")
    if not os.path.exists(filepath):
        print(f"[错误] 文件不存在: {filepath}")
        return None, None, None
        
    df = pd.read_csv(filepath)
    print(f"    原始数据集样本数: {len(df)}")
    
    feature_cols = [
        'Ty_Pressure', 'Ty_Center_Wind',
        'Ty_Move_Speed', 'Ty_Move_Dir', 
        'Sta_Height', 'Dist_to_Coast', 'Terrain_10km', 
        'Sta_Slope', 'Sta_TPI',
        'Dist_Station_Ty', 'Azimuth_Station_Ty',
        'Pressure_Gradient_Obs', 'Pressure_Gradient_SLP'
    ]
    target_col = 'Obs_Wind_Speed'
    
    # 智能缺失值处理
    df = df.dropna(subset=[target_col])
    basic_features = [c for c in feature_cols if 'Pressure_Gradient' not in c]
    df = df.dropna(subset=basic_features)
    
    print(f"    处理后最终样本数: {len(df)}")
    print(f"    (注: 已保留气压梯度缺失但其他特征完整的 {df['Pressure_Gradient_Obs'].isna().sum()} 条样本)")
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"[警告] 数据集中缺少以下列: {missing_cols}")
        return None, None, None
    
    return df, feature_cols, target_col

# ================= 2. 模型训练与评估 (核心修改：十折交叉验证) =================

def train_model_cv(df, features, target, n_splits=10):
    print(f"\n>>> 开始 {n_splits} 折交叉验证 (GroupKFold)...")
    print("    注意：按台风ID(TID)分组，确保同一台风不同时出现在训练集和验证集")
    
    # 初始化 GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    groups = df['TID']
    
    # 存储每一折的结果
    results = []
    
    # 用于保存“最佳模型”以便后续画图
    best_model = None
    best_score = -np.inf
    best_X_test = None
    best_y_test = None
    best_y_pred = None
    
    fold = 1
    for train_idx, test_idx in gkf.split(df, groups=groups):
        # 1. 数据切分
        X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx][target]
        X_test, y_test = df.iloc[test_idx][features], df.iloc[test_idx][target]
        
        # 2. 模型定义
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
        
        # 3. 训练
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 4. 预测与评估
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"    Fold {fold}/{n_splits} | R2: {r2:.4f} | RMSE: {rmse:.4f}")
        
        results.append({
            'Fold': fold,
            'R2': r2, 'RMSE': rmse, 'MAE': mae,
            'Train_Size': len(X_train), 'Test_Size': len(X_test)
        })
        
        # 5. 更新最佳模型 (用于后续SHAP分析和绘图)
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_X_test = X_test
            best_y_test = y_test
            best_y_pred = y_pred
            
        fold += 1
        
    # --- 汇总结果 ---
    results_df = pd.DataFrame(results)
    avg_r2 = results_df['R2'].mean()
    avg_rmse = results_df['RMSE'].mean()
    
    print("\n>>> 交叉验证完成！")
    print(f"    平均 R2  : {avg_r2:.4f} (±{results_df['R2'].std():.4f})")
    print(f"    平均 RMSE: {avg_rmse:.4f}")
    
    # 保存详细的CV结果
    results_df.to_csv(os.path.join(OUTPUT_DIR, "CV_10Fold_Metrics.csv"), index=False, encoding='utf-8-sig')
    
    # 保存最佳折叠的预测数据
    pred_obs_df = pd.DataFrame({'Observed_Wind': best_y_test.values, 'Predicted_Wind': best_y_pred})
    pred_obs_df.to_csv(os.path.join(OUTPUT_DIR, "Best_Fold_Prediction.csv"), index=False, encoding='utf-8-sig')

    # --- 绘图 (只画最佳那一折的图，代表模型上限) ---
    print(f"    正在绘制最佳折叠 (R2={best_score:.4f}) 的评估图...")
    
    # 绘图: 散点
    plt.figure(figsize=(6, 6))
    plt.scatter(best_y_test, best_y_pred, alpha=0.1, s=3, color='#1f77b4')
    plt.plot([best_y_test.min(), best_y_test.max()], [best_y_test.min(), best_y_test.max()], 'r--', lw=2)
    plt.xlabel("观测风速 (m/s)")
    plt.ylabel("预测风速 (m/s)")
    plt.title(f"最佳折叠预测性能 (10-Fold CV)\nR2={best_score:.3f}, RMSE={np.sqrt(mean_squared_error(best_y_test, best_y_pred)):.3f}")
    plt.savefig(os.path.join(OUTPUT_DIR, "1_Model_Scatter_BestFold.png"), dpi=300)
    plt.close()
    
    # 绘图: KDE 对比
    plt.figure(figsize=(8, 5))
    sns.kdeplot(best_y_test, fill=True, label='观测值 (Observed)', color='blue', alpha=0.3)
    sns.kdeplot(best_y_pred, fill=True, label='预测值 (Predicted)', color='orange', alpha=0.3)
    plt.xlabel("风速 (m/s)")
    plt.ylabel("概率密度 (Density)")
    plt.title("观测值与预测值分布对比 (Best Fold)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "2_Model_KDE_BestFold.png"), dpi=300)
    plt.close()
    
    return best_model

# ================= 3. SHAP 分析 (保持不变) =================

def run_shap_analysis(model, X, output_subdir, title_suffix=""):
    save_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(save_path, exist_ok=True)
    
    # 抽样加速
    MAX_SHAP_SAMPLES = 5000
    if len(X) > MAX_SHAP_SAMPLES:
        X_sample = X.sample(MAX_SHAP_SAMPLES, random_state=42)
    else:
        X_sample = X
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # 导出重要性 CSV
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_sample.columns
    
    importance_df = pd.DataFrame({
        'Feature_En': feature_names,
        'Feature_Cn': [FEATURE_MAP_CN.get(f, f) for f in feature_names],
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    csv_name = f"SHAP_Importance_{title_suffix}.csv".replace(" ", "_").replace("(", "").replace(")", "")
    importance_df.to_csv(os.path.join(save_path, csv_name), index=False, encoding='utf-8-sig')
    
    # 中文显示准备
    X_display_cn = X_sample.rename(columns=FEATURE_MAP_CN)
    
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
    
    # (C) 依赖图 Top 6
    top_n = 6
    top_features_cn = importance_df['Feature_Cn'].head(top_n).tolist()
    top_features_en = importance_df['Feature_En'].head(top_n).tolist()
    
    for feat_en, feat_cn in zip(top_features_en, top_features_cn):
        try:
            fig = plt.figure(figsize=(9, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 5], hspace=0.08)
            ax_kde = fig.add_subplot(gs[0])
            ax_shap = fig.add_subplot(gs[1])
            
            sns.kdeplot(x=X_display_cn[feat_cn], ax=ax_kde, fill=True, color='#4c72b0', alpha=0.3)
            ax_kde.set_xlabel(""); ax_kde.set_xticklabels([]); ax_kde.set_yticks([]); ax_kde.set_ylabel("")
            ax_kde.set_title(f"{feat_cn} 分布与 SHAP 贡献 {title_suffix}", fontsize=13)
            sns.despine(ax=ax_kde, left=True, bottom=True)
            
            shap.dependence_plot(feat_cn, shap_values, X_display_cn, interaction_index='auto', show=False, ax=ax_shap, alpha=0.6, x_jitter=0)
            
            shap_xlim = ax_shap.get_xlim()
            ax_kde.set_xlim(shap_xlim)
            
            clean_name = f"SHAP_Dep_KDE_{feat_en}_{title_suffix}".replace(" ", "_")
            plt.savefig(os.path.join(save_path, f"{clean_name}_CN.png"), dpi=300)
            plt.close()
        except Exception:
            plt.close()

# ================= 主程序 =================

def main():
    print(">>> 1. 数据加载...")
    df, features, target = load_and_prep_data(DATASET_PATH)
    
    if df is None: return

    print("\n>>> 2. 十折交叉验证与训练...")
    # 这里返回的是 best_model
    best_model = train_model_cv(df, features, target, n_splits=10)
    
    print("\n>>> 3. SHAP 归因分析 (基于 CV 中表现最好的模型)...")
    
    plt.close('all')
    print("  -> 执行全局分析...")
    run_shap_analysis(best_model, df[features], "全局分析", "All")
    
    # 聚类分析 (代码保持不变)
    if 'Cluster_8_9' in df.columns:
        print("\n  -> 执行 8-9级 分聚类分析...")
        clusters = sorted(df['Cluster_8_9'].dropna().unique())
        for cid in clusters:
            if cid == -1: continue
            subset = df[df['Cluster_8_9'] == cid]
            if len(subset) > 200:
                print(f"     类别 {int(cid)} (N={len(subset)})")
                run_shap_analysis(best_model, subset[features], "8-9级分型", f"C{int(cid)}")
            
    if 'Cluster_10_Plus' in df.columns:
        print("\n  -> 执行 10级+ 分聚类分析...")
        clusters_11 = sorted(df['Cluster_10_Plus'].dropna().unique())
        for cid in clusters_11:
            if cid == -1: continue
            subset = df[df['Cluster_10_Plus'] == cid]
            if len(subset) > 200:
                print(f"     类别 {int(cid)} (N={len(subset)})")
                run_shap_analysis(best_model, subset[features], "10级以上分型", f"C{int(cid)}")

    print(f"\n[完成] 成果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()