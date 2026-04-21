# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

class AdaptiveTailProcessor:
    def __init__(self, df, feature_cols):
        self.df = df
        self.feature_cols = feature_cols
        self.stats_report = {}
        self.feature_strategies = {}
        self.scalers = {}

    def analyze_distributions(self):
        logger.info(">>> [Analysis] Starting numerical analysis of feature distributions...")
        for col in self.feature_cols:
            data = self.df[col].dropna().astype(float)
            if len(data) == 0: continue

            skewness = data.skew()
            p99 = data.quantile(0.99)
            max_val = data.max()
            
            # Tail-Fatness Ratio: Max vs 99th percentile
            tail_ratio = max_val / p99 if p99 > 0 else 1.0

            self.stats_report[col] = {
                'skewness': skewness,
                'tail_ratio': tail_ratio,
            }
            logger.info(f"Feature: {col:<25} | Skewness: {skewness:>8.2f} | Tail-Ratio: {tail_ratio:>8.2f}")

    def design_strategy(self):
        logger.info(">>> [Strategy] Designing optimal processing strategies...")
        for col, stats in self.stats_report.items():
            skew = abs(stats['skewness'])
            tail_ratio = stats['tail_ratio']

            if skew > 50 or tail_ratio > 10:
                self.feature_strategies[col] = 'EXTREME_TAIL'
            elif skew > 5 or tail_ratio > 3:
                self.feature_strategies[col] = 'MODERATE_SKEW'
            else:
                self.feature_strategies[col] = 'NORMAL'
            logger.info(f"Assigned Strategy -> {col:<25} : {self.feature_strategies[col]}")

    def transform_features(self):
        logger.info(">>> [Transform] Applying adaptive transformations...")
        transformed_df = pd.DataFrame(index=self.df.index)

        for col in self.feature_cols:
            strategy = self.feature_strategies.get(col, 'NORMAL')
            raw_data = self.df[col].clip(lower=0).values.reshape(-1, 1)

            if strategy == 'EXTREME_TAIL':
                log_data = np.log1p(raw_data)
                scaler = RobustScaler(quantile_range=(1.0, 99.0))
                transformed_df[col] = scaler.fit_transform(log_data).flatten()
                self.scalers[col] = {'type': 'log_robust', 'scaler': scaler}
            elif strategy == 'MODERATE_SKEW':
                log_data = np.log1p(raw_data)
                scaler = StandardScaler()
                transformed_df[col] = scaler.fit_transform(log_data).flatten()
                self.scalers[col] = {'type': 'log_standard', 'scaler': scaler}
            else:
                scaler = StandardScaler()
                transformed_df[col] = scaler.fit_transform(raw_data).flatten()
                self.scalers[col] = {'type': 'standard', 'scaler': scaler}

        return transformed_df.values

    def calculate_adaptive_weights(self, transformed_data):
        logger.info(">>> [Weighting] Calculating density-inverse sampling weights...")
        extreme_cols_idx = [i for i, col in enumerate(self.feature_cols) if self.feature_strategies.get(col) == 'EXTREME_TAIL']
        
        if not extreme_cols_idx:
            logger.info("No extreme tail features found. Using uniform weights.")
            return torch.DoubleTensor(np.ones(len(transformed_data)))

        tail_features = transformed_data[:, extreme_cols_idx]
        binned_feats = np.zeros_like(tail_features, dtype=int)
        
        for i in range(tail_features.shape[1]):
            bins = np.percentile(tail_features[:, i], np.linspace(0, 100, 11))
            bins[-1] += 1e-5 
            binned_feats[:, i] = np.digitize(tail_features[:, i], bins)
            
        grid_keys = ['_'.join(map(str, row)) for row in binned_feats]
        keys_series = pd.Series(grid_keys)
        grid_counts = keys_series.value_counts().to_dict()
        
        # Inverse density weighting (with sqrt dampening)
        weights = keys_series.apply(lambda k: 1.0 / np.sqrt(grid_counts[k])).values
        return torch.DoubleTensor(weights)

    def process(self):
        self.analyze_distributions()
        self.design_strategy()
        transformed_data = self.transform_features()
        sampling_weights = self.calculate_adaptive_weights(transformed_data)
        return transformed_data, sampling_weights, self.scalers