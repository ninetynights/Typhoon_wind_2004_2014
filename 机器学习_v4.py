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
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 
sns.set_style("whitegrid", {"font.sans-serif": ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']})

# ================= 配置区域 =================
BASE_DIR = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024"

# [修改点1] 更新为 v3 版本数据集 (包含气压梯度特征)
DATASET_PATH = os.path.join(BASE_DIR, "输出_机器学习", "Final_Training_Dataset_XGBoost_v3_Pressure.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "输出_机器学习", "模型与SHAP分析结果_聚类结果2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 特征中英文映射字典 ---
FEATURE_MAP_CN = {
    # 原有特征
    'Ty_Pressure': '台风中心气压',
    'Ty_Center_Wind': '台风近中心风速',
    'Sta_Height': '站点海拔',
    'Dist_to_Coast': '离岸距离',
    'Terrain_10km': '地形开阔度(10km)',
    'Dist_Station_Ty': '站点与台风距离',
    'Azimuth_Station_Ty': '站点相对方位角',
    
    # 新增特征 (v2)
    'Ty_Move_Speed': '台风移动速度',
    'Ty_Move_Dir': '台风移动方向',
    'Sta_Slope': '站点坡度',
    'Sta_TPI': '地形位置指数',
    
    # [修改点2] 新增气压梯度特征 (v3)
    'Pressure_Gradient_Obs': '观测气压梯度(Obs)',
    'Pressure_Gradient_SLP': '海平面气压梯度(SLP)'
}

# ================= 1. 数据准备 =================
def load_and_prep_data(filepath):
    print(f">>> 正在读取数据集: {filepath}")
    if not os.path.exists(filepath):
        print(f"[错误] 文件不存在: {filepath}")
        return None, None, None
        
    df = pd.read_csv(filepath)
    print(f"    原始数据集样本数: {len(df)}")
    
    # ---------------------------------------------------------
    # 1. 定义特征列表
    # ---------------------------------------------------------
    feature_cols = [
        # --- 台风属性 ---
        'Ty_Pressure', 'Ty_Center_Wind',
        'Ty_Move_Speed', 'Ty_Move_Dir', 
        
        # --- 站点静态属性 ---
        'Sta_Height', 'Dist_to_Coast', 'Terrain_10km', 
        'Sta_Slope', 'Sta_TPI',
        
        # --- 相对位置属性 ---
        'Dist_Station_Ty', 'Azimuth_Station_Ty',
        
        # --- 气压梯度属性 (允许缺失) ---
        'Pressure_Gradient_Obs', 'Pressure_Gradient_SLP'
    ]
    target_col = 'Obs_Wind_Speed'
    
    # ---------------------------------------------------------
    # 2. 智能缺失值处理 (核心修改)
    # ---------------------------------------------------------
    
    # 第一步：必须删除 目标变量 (Y) 缺失的行
    # (没有真值无法训练)
    df = df.dropna(subset=[target_col])
    
    # 第二步：必须删除 基础特征 (X_basic) 缺失的行
    # (经纬度、距离、地形等如果缺失，通常是数据错误，必须删)
    # 我们排除掉那两个气压梯度特征，只检查其他特征
    basic_features = [c for c in feature_cols if 'Pressure_Gradient' not in c]
    df = df.dropna(subset=basic_features)
    
    # 第三步：气压梯度特征 (X_pressure) 保持原样
    # XGBoost 原生支持 NaN，它会自动学习"没有气压数据"代表什么物理含义
    # (通常会被分到某一个特定的子节点)
    
    print(f"    处理后最终样本数: {len(df)}")
    print(f"    (注: 已保留气压梯度缺失但其他特征完整的 {df['Pressure_Gradient_Obs'].isna().sum()} 条样本)")
    
    # ---------------------------------------------------------
    # 3. 完整性检查
    # ---------------------------------------------------------
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"[警告] 数据集中缺少以下列: {missing_cols}")
        print("请检查是否使用了正确版本的 CSV 文件")
        return None, None, None
    
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
    
    print("\n>>> 开始训练 XGBoost ")
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

# ================= 3. SHAP 分析 =================

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
    
    # (C) 依赖图 + 顶部 KDE 分布
    # 挑选 Top 6 特征 (增加特征后多看几个，尤其是新加的气压特征很可能排前面)
    top_n = 6
    top_features_cn = importance_df['Feature_Cn'].head(top_n).tolist()
    top_features_en = importance_df['Feature_En'].head(top_n).tolist()
    
    for feat_en, feat_cn in zip(top_features_en, top_features_cn):
        try:
            fig = plt.figure(figsize=(9, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 5], hspace=0.08)
            
            ax_kde = fig.add_subplot(gs[0])
            ax_shap = fig.add_subplot(gs[1])
            
            # 1. 绘制上方 KDE 图
            sns.kdeplot(
                x=X_display_cn[feat_cn], 
                ax=ax_kde, 
                fill=True, 
                color='#4c72b0', 
                alpha=0.3, 
                linewidth=1.5
            )
            ax_kde.set_xlabel("")
            ax_kde.set_xticklabels([])
            ax_kde.set_yticks([])
            ax_kde.set_ylabel("")
            ax_kde.set_title(f"{feat_cn} 分布与 SHAP 贡献分析 {title_suffix}", fontsize=13, pad=10)
            sns.despine(ax=ax_kde, left=True, bottom=True)
            
            # 2. 绘制下方 SHAP 依赖图
            shap.dependence_plot(
                feat_cn, 
                shap_values, 
                X_display_cn, 
                interaction_index='auto', 
                show=False,
                ax=ax_shap,
                alpha=0.6,
                x_jitter=0
            )
            
            shap_xlim = ax_shap.get_xlim()
            ax_kde.set_xlim(shap_xlim)
            ax_shap.set_ylabel("SHAP值 (对风速的贡献 m/s)")
            
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
    
    if df is None:
        print("[终止] 数据加载失败")
        return

    print("\n>>> 2. 模型训练与评估...")
    model = train_model(df, features, target)
    
    print("\n>>> 3. SHAP 归因分析 (带 KDE 分布图)...")
    
    plt.close('all')
    
    print("  -> 执行全局分析...")
    run_shap_analysis(model, df[features], "全局分析", "All")
    
    # 检查并执行 8-9级 聚类分析 (修正后的列名)
    if 'Cluster_8_9' in df.columns:
        print("\n  -> 执行 8-9级 分聚类分析...")
        clusters = sorted(df['Cluster_8_9'].dropna().unique())
        for cid in clusters:
            if cid == -1: continue
            subset = df[df['Cluster_8_9'] == cid]
            if len(subset) > 200:
                print(f"     类别 {int(cid)} (N={len(subset)})")
                run_shap_analysis(model, subset[features], "8-9级分型", f"C{int(cid)}")
    else:
        print("\n[警告] 未找到 'Cluster_8_9' 列，跳过 8-9 级分析。")
            
    # 检查并执行 10级+ 聚类分析 (修正后的列名)
    if 'Cluster_10_Plus' in df.columns:
        print("\n  -> 执行 10级+ 分聚类分析...")
        clusters_11 = sorted(df['Cluster_10_Plus'].dropna().unique())
        for cid in clusters_11:
            if cid == -1: continue
            subset = df[df['Cluster_10_Plus'] == cid]
            if len(subset) > 200:
                print(f"     类别 {int(cid)} (N={len(subset)})")
                run_shap_analysis(model, subset[features], "10级以上分型", f"C{int(cid)}")
    else:
        print("\n[警告] 未找到 'Cluster_10_Plus' 列，跳过 10级+ 分析。")

    print(f"\n[完成] 成果已保存至: {OUTPUT_DIR}")
    print("包含特征: 基础气象 + 静态地理(含TPI/Slope) + 动态路径(含移速/向) + 气压梯度")

if __name__ == "__main__":
    main()