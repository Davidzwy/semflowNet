# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import joblib
from sentence_transformers import SentenceTransformer

from model import FlowDriveNet, train_flowdrive
from train_handler import AdaptiveTailProcessor # Import our new pipeline

# 导入你的自定义模块
from model import FlowDriveNet, train_flowdrive
from train_handler import AdaptiveTailProcessor 

# ================= 配置与路径 =================
REAL_DATA_PATH = '/home/zwy/TrajFlow/ori2_48.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256

# --- 断点续训配置 ---
RESUME_TRAINING = True  # 是否从断点继续
# 指向你之前保存的 300 epoch 模型
PRETRAINED_MODEL_PATH = '/home/zwy/flowdrive/training_resume_20260328_021717/flow_model_epoch_900.pth'
START_EPOCH = 900       # 起始轮数
TOTAL_EPOCHS = 1800      # 目标总轮数
SAVE_INTERVAL = 50

# --- 输出目录 ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f'./training_resume_{timestamp}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALER_PATH = os.path.join(OUTPUT_DIR, 'scalers_dict.pkl')
FLOW_MODEL_PATH = os.path.join(OUTPUT_DIR, 'flow_model_final.pth')
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'resume_training.log')

# ================= 日志配置 =================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger()

class FlowDriveDataset(Dataset):
    def __init__(self, df, transformed_data, weights, text_model_name='all-MiniLM-L6-v2'): 
        self.continuous_features = transformed_data
        self.weights = weights 

        logger.info(">>> Generating semantic text prompts for flows...")
        prompts = df.apply(
            lambda row: f"A {row['category']} network flow using {row['application_protocol']} protocol "
                        f"on port {row['src_port']} to {row['dst_port']}, providing {row['web_service']} service.",
            axis=1
        ).tolist()

        logger.info(f">>> Extracting LLM Semantic Embeddings using {text_model_name}...")
        llm_model = SentenceTransformer(text_model_name, device=DEVICE)
        self.semantic_embeddings = llm_model.encode(prompts, show_progress_bar=True, batch_size=256)
        self.semantic_embeddings = torch.tensor(self.semantic_embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.continuous_features)

    def __getitem__(self, idx):
        return {
            'continuous_features': torch.tensor(self.continuous_features[idx], dtype=torch.float32),
            'semantic_emb': self.semantic_embeddings[idx],
            'weight': torch.tensor(self.weights[idx], dtype=torch.float32) 
        }

def main():
    logger.info(f"Resume session. Outputs: {OUTPUT_DIR}")
    
    # 1. 加载数据
    logger.info(">>> 1. Loading data...")
    df_real = pd.read_csv(REAL_DATA_PATH).fillna(0)

    ignore_cols = ['flow_key', 'src_ip', 'dst_ip', 'src_ip_numeric', 'src_port', 'dst_port', 'proto',
                   'flowStart', 'flowEnd', 'f_flowStart', 'f_flowEnd', 'b_flowStart', 'b_flowEnd',
                   'flowEndReason', 'category', 'application_protocol', 'web_service', 'target_goal_idx']
    feature_cols = [col for col in df_real.columns if col not in ignore_cols]

    # 2. 预处理 (注意：继续训练时应确保 Processor 的逻辑与之前一致)
    logger.info(">>> 2. Adaptive Tail-Aware Processing...")
    processor = AdaptiveTailProcessor(df_real, feature_cols)
    transformed_data, weights, scalers = processor.process()
    joblib.dump(scalers, SCALER_PATH)

    # 3. 数据加载器
    logger.info(">>> 3. Preparing Balanced DataLoader...")
    dataset = FlowDriveDataset(df_real, transformed_data, weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # 4. 初始化模型
    logger.info(">>> 4. Initializing FlowDriveNet...")
    flow_model = FlowDriveNet(num_features=len(feature_cols), semantic_dim=384, hidden_dim=512).to(DEVICE)

    # --- 核心：加载断点权重 ---
    current_start = 0
    if RESUME_TRAINING:
        if os.path.exists(PRETRAINED_MODEL_PATH):
            logger.info(f"? Loading pretrained model from {PRETRAINED_MODEL_PATH}")
            flow_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
            current_start = START_EPOCH
        else:
            logger.error(f"? Pretrained file not found at {PRETRAINED_MODEL_PATH}. Starting from 0.")

    # 5. 开始训练
    logger.info(f">>> 5. Training from Epoch {current_start} to {TOTAL_EPOCHS}...")
    trained_flow = train_flowdrive(
        dataloader, 
        flow_model, 
        epochs=TOTAL_EPOCHS, 
        device=DEVICE,
        save_dir=OUTPUT_DIR, 
        save_interval=SAVE_INTERVAL,
        start_epoch=current_start  # 确保你的 model.py 支持此参数
    )

    # 6. 保存最终结果
    torch.save(trained_flow.state_dict(), FLOW_MODEL_PATH)
    logger.info(f"Training finished. Final model: {FLOW_MODEL_PATH}")

if __name__ == "__main__":
    main()