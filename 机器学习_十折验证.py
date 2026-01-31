import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================= 配置 =================
# 读取您刚刚跑出来的最佳折叠预测结果
FILE_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风_2004_2024/输出_机器学习/模型与SHAP分析结果_10折验证版/Best_Fold_Prediction.csv"

def evaluate_subset(df, subset_name):
    """计算子集的评估指标"""
    if len(df) == 0:
        print(f"--- {subset_name}: 样本数为0，跳过 ---")
        return
    
    y_true = df['Observed_Wind']
    y_pred = df['Predicted_Wind']
    
    # 基础指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # [关键] 偏差 Bias (预测 - 实况)
    #如果 < 0 说明平均来看是“报小了”
    bias = np.mean(y_pred - y_true)
    
    # [关键] 准确率 (基于风级)
    # 简单估算：误差在 ±2.5m/s (约1级) 以内的比例
    accuracy_1_grade = np.mean(np.abs(y_pred - y_true) <= 2.5) * 100
    
    print(f"{subset_name:<15} | 样本数: {len(df):<5} | R2: {r2:6.3f} | RMSE: {rmse:5.2f} | Bias: {bias:5.2f} | ±1级准确率: {accuracy_1_grade:5.1f}%")

# ================= 主程序 =================
try:
    df = pd.read_csv(FILE_PATH)
    print(f"已加载预测结果，总样本: {len(df)}")
    print("-" * 90)
    print(f"{'分段名称':<15} | {'样本数':<5} | {'R2':<6} | {'RMSE':<5} | {'Bias':<5} | {'±1级准确率'}")
    print("-" * 90)

    # 1. 全体数据
    evaluate_subset(df, "全体数据 (All)")

    # 2. 8-9级风 (17.2 <= 风速 < 24.5)
    # 对应 Beaufort Scale 定义
    mask_8_9 = (df['Observed_Wind'] >= 17.2) & (df['Observed_Wind'] < 24.5)
    evaluate_subset(df[mask_8_9], "8-9级 (一般)")

    # 3. 10级及以上 (风速 >= 24.5)
    mask_10_plus = (df['Observed_Wind'] >= 24.5)
    evaluate_subset(df[mask_10_plus], "10级+ (极端)")
    
    # 4. 12级及以上 (风速 >= 32.7) - 看看极端的表现
    mask_12_plus = (df['Observed_Wind'] >= 32.7)
    evaluate_subset(df[mask_12_plus], "12级+ (核心)")

    print("-" * 90)
    print("注：Bias < 0 表示模型倾向于低估风速（报小了）；Bias > 0 表示高估（报大了）。")

except FileNotFoundError:
    print(f"[错误] 找不到文件: {FILE_PATH}")
    print("请确认您是否已经运行了 '机器学习_v5.py' 并且生成了 Best_Fold_Prediction.csv")