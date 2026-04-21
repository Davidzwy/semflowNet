# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler
import os

REAL_CSV_PATH = '/home/zwy/TrajFlow/ori2_48.csv'
# 这里填你原本没有经过任何 mixup 或 cdf 对齐的原始生成数据！
GEN_CSV_PATH = '/home/zwy/flowdrive/final_synthetic_goalflow_dataset.csv' 

def main():
    print(">>> 1. Loading Datasets for Diagnostic Analysis...")
    df_real = pd.read_csv(REAL_CSV_PATH).fillna(0)
    df_gen = pd.read_csv(GEN_CSV_PATH).fillna(0)
    
    ignore_cols = [
        'flow_key', 'src_ip', 'dst_ip', 'src_ip_numeric', 'src_port', 'dst_port', 'proto',
        'flowStart', 'flowEnd', 'f_flowStart', 'f_flowEnd', 'b_flowStart', 'b_flowEnd',
        'flowEndReason', 'category', 'application_protocol', 'web_service', 'target_goal_idx'
    ]
    
    features = [c for c in df_gen.columns if c in df_real.columns and c not in ignore_cols]
    features = [c for c in features if pd.api.types.is_numeric_dtype(df_real[c])]
    
    print(f">>> Analyzing {len(features)} numerical features...\n")
    
    # 为了让 Wasserstein 距离在不同量级的特征间可比，我们先在真实数据上 fit 一个 scaler
    scaler = StandardScaler()
    scaler.fit(df_real[features])
    
    real_scaled = pd.DataFrame(scaler.transform(df_real[features]), columns=features)
    gen_scaled = pd.DataFrame(scaler.transform(df_gen[features]), columns=features)
    
    report = []
    
    for col in features:
        # 取出真实和生成列 (归一化前用于 KS, 归一化后用于 Wasserstein)
        r_raw = df_real[col].values
        g_raw = df_gen[col].values
        
        r_scaled = real_scaled[col].values
        g_scaled = gen_scaled[col].values
        
        # 1. 基础统计量差异
        mean_diff_pct = np.abs(r_raw.mean() - g_raw.mean()) / (np.abs(r_raw.mean()) + 1e-9)
        std_diff_pct = np.abs(r_raw.std() - g_raw.std()) / (np.abs(r_raw.std()) + 1e-9)
        
        # 2. KS Test (最大分布函数差)
        ks_stat, _ = ks_2samp(r_raw, g_raw)
        
        # 3. Wasserstein Distance (形状差异)
        wd = wasserstein_distance(r_scaled, g_scaled)
        
        report.append({
            'Feature': col,
            'KS_Stat': ks_stat,
            'Wasserstein': wd,
            'Mean_Drift_Ratio': mean_diff_pct,
            'Std_Drift_Ratio': std_diff_pct
        })
        
    df_report = pd.DataFrame(report)
    
    # 按照 KS 统计量降序排列（最严重的特征排在前面）
    df_report = df_report.sort_values(by='KS_Stat', ascending=False).reset_index(drop=True)
    
    print("=" * 80)
    print("? TOP 10 MOST DRIFTED FEATURES (Worst Performers)")
    print("=" * 80)
    print(df_report.head(10).to_string(index=False, float_format="%.4f"))
    print("\n")
    
    # 4. 分析相关性矩阵的破坏程度
    corr_real = df_real[features].corr().fillna(0).values
    corr_gen = df_gen[features].corr().fillna(0).values
    corr_diff = np.abs(corr_real - corr_gen)
    
    print("=" * 80)
    print(f"? Feature Correlation Drift (Mean Absolute Error): {corr_diff.mean():.4f}")
    print("=" * 80)

    # 保存完整报告
    df_report.to_csv('feature_drift_report.csv', index=False)
    print("\n>>> Full diagnostic report saved to 'feature_drift_report.csv'")

if __name__ == "__main__":
    main()