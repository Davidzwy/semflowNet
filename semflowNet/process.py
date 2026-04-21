# # -*- coding: utf-8 -*-
# import pandas as pd
# import numpy as np
# import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# import torch
# import joblib
# import logging
# from sentence_transformers import SentenceTransformer

# # 导入你最新包含 Cross-Attention 的模型
# from model import FlowDriveNet

# # ================= 配置路径 =================
# REAL_DATA_PATH = '/home/zwy/TrajFlow/ori2_48.csv'
# OUTPUT_FILE = 'final_synthetic_goalflow_dataset.csv'

# # 请确保这些路径指向你 train.py 生成的最新时间戳文件夹
# OUTPUT_DIR = '/home/zwy/flowdrive/training_resume_20260407_111733'
# SCALER_PATH = os.path.join(OUTPUT_DIR, 'scalers_dict.pkl') 
# FLOW_MODEL_PATH = os.path.join(OUTPUT_DIR, 'flow_model_epoch_300.pth')

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger()
# # ============================================

# @torch.no_grad()
# def generate_moderated_flow(flow_model, semantic_emb, num_features, num_steps=10, cfg_scale=2.5, sde_noise_scale=0.2):

#     flow_model.eval()
#     batch_size = semantic_emb.shape[0]

#     # 在隐空间采样纯高斯噪声 z_0
#     latent_dim = flow_model.flow.out_layer[-1].out_features
#     z_t = torch.randn(batch_size, latent_dim).to(DEVICE)
#     dt = 1.0 / num_steps

#     uncond_emb = torch.zeros_like(semantic_emb).to(DEVICE)

#     for step in range(num_steps):
#         t = torch.full((batch_size,), step * dt).to(DEVICE)

#         # 1. 预测速度场 (结合无分类器引导 CFG)
#         if cfg_scale != 1.0:
#             z_in = torch.cat([z_t, z_t], dim=0)
#             t_in = torch.cat([t, t], dim=0)
#             emb_in = torch.cat([uncond_emb, semantic_emb], dim=0)

#             # 调用 forward_flow 在隐空间预测速度场
#             v_pred = flow_model.forward_flow(z_in, t_in, emb_in)
#             v_uncond, v_cond = v_pred.chunk(2, dim=0)
#             v_t_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
#         else:
#             v_t_pred = flow_model.forward_flow(z_t, t, semantic_emb)

#         # 2. 【核心前沿优化】：SDE 随机探索与自我纠错机制
#         if step < num_steps - 1:
#             # 注入与时间步长 dt 相关的布朗运动噪声 (Langevin Dynamics)
#             noise = torch.randn_like(z_t) * np.sqrt(dt) * sde_noise_scale
#         else:
#             # 最后一步必须是纯净的确定性步进，消除最终的白噪声残留
#             noise = 0

#         # 3. 欧拉更新 + 随机微调
#         z_t = z_t + v_t_pred * dt + noise

#     # 生成完成后，通过解码器无损还原为原始的 48 维物理特征
#     x_generated = flow_model.decode(z_t)
#     return x_generated


# def main():
#     logger.info(">>> 1. Loading Models and Assets...")
#     df_real = pd.read_csv(REAL_DATA_PATH).fillna(0)

#     ignore_cols = ['flow_key', 'src_ip', 'dst_ip', 'src_ip_numeric', 'src_port', 'dst_port', 'proto',
#                    'flowStart', 'flowEnd', 'f_flowStart', 'f_flowEnd', 'b_flowStart', 'b_flowEnd',
#                    'flowEndReason', 'category', 'application_protocol', 'web_service', 'target_goal_idx']
    
#     # 获取连续特征列名
#     feature_cols = [col for col in df_real.columns if col not in ignore_cols]
#     num_features = len(feature_cols)

#     # 加载 AdaptiveTailProcessor 保存的复杂 Scaler 字典
#     scalers = joblib.load(SCALER_PATH)
    
#     # 加载 LLM 提取语义
#     llm_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

#     # 初始化带有 Cross-Attention 的流匹配模型
#     flow_model = FlowDriveNet(num_features=num_features, semantic_dim=384, hidden_dim=512).to(DEVICE)
#     flow_model.load_state_dict(torch.load(FLOW_MODEL_PATH, map_location=DEVICE))

#     if os.path.exists(OUTPUT_FILE):
#         os.remove(OUTPUT_FILE)

#     write_header = True
#     total_gen = 0

#     logger.info(">>> 2. Generating High-Fidelity Diverse Traffic...")
    
#     batch_size = 500
#     num_samples = len(df_real)
    
#     # 分批次生成，防止显存/内存溢出
#     for start_idx in range(0, num_samples, batch_size):
#         # 截取真实数据的上下文作为生成条件
#         sampled_context = df_real.sample(n=min(batch_size, num_samples - start_idx), replace=True).copy()
        
#         # 重建与 train.py 完全一致的自然语言 Prompt
#         prompts = sampled_context.apply(
#             lambda row: f"A {row['category']} network flow using {row['application_protocol']} protocol "
#                         f"on port {row['src_port']} to {row['dst_port']}, providing {row['web_service']} service.", 
#             axis=1
#         ).tolist()
        
#         # 提取语义特征向量 [B, 384]
#         semantic_embeddings = llm_model.encode(prompts, show_progress_bar=False)
#         t_semantic_emb = torch.tensor(semantic_embeddings, dtype=torch.float32).to(DEVICE)

#         # =========================================================
#         # 核心：生成归一化后的连续特征
#         # =========================================================
#         generated_normalized = generate_moderated_flow(
#             flow_model, 
#             t_semantic_emb, 
#             num_features, 
#             num_steps=20,
#             cfg_scale=1.2,       # CFG 引导强度
#             sde_noise_scale=0.05  # 【新增】SDE 噪声强度
#         ).cpu().numpy()

#         df_gen = pd.DataFrame(columns=feature_cols)

#         # ---------------------------------------------------------
#         # 根据 AdaptiveTailProcessor 的策略自适应逆变换
#         # ---------------------------------------------------------
#         for i, col in enumerate(feature_cols):
#             strat = scalers[col]['type']
#             scaler = scalers[col]['scaler']
#             col_data = generated_normalized[:, i].reshape(-1, 1)
            
#             # 执行逆标准化/逆稳健缩放
#             inv_data = scaler.inverse_transform(col_data).flatten()
            
#             # 如果训练时用了 log1p，这里用 expm1 还原
#             if strat in ['log_robust', 'log_standard']:
#                 inv_data = np.expm1(inv_data) 
                
#             df_gen[col] = inv_data

#         # ---------------------------------------------------------
#         # 还原物理约束与离散上下文
#         # ---------------------------------------------------------
#         for c in ['src_port', 'dst_port', 'proto', 'category', 'application_protocol', 'web_service']:
#             df_gen[c] = sampled_context[c].values

#         # 1. 物理裁剪 (包数、字节数不能小于0)
#         df_gen[feature_cols] = df_gen[feature_cols].clip(lower=0)

#         # 2. 逻辑一致性校验
#         if 'f_pktTotalCount' in df_gen.columns:
#             df_gen['pktTotalCount'] = df_gen['f_pktTotalCount'] + df_gen['b_pktTotalCount']
#             df_gen['octetTotalCount'] = df_gen['f_octetTotalCount'] + df_gen['b_octetTotalCount']
#         if 'max_ps' in df_gen.columns:
#             df_gen['max_ps'] = df_gen[['max_ps', 'min_ps']].max(axis=1)

#         # 3. 数据类型修正
#         for col in df_gen.columns:
#             if 'Count' in col or 'port' in col or 'proto' in col or 'ps' in col:
#                 if pd.api.types.is_numeric_dtype(df_gen[col]):
#                     df_gen[col] = df_gen[col].round().astype(int)
#         # ---------------------------------------------------------

#         # 追加写入 CSV
#         df_gen.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)
#         write_header = False
#         total_gen += len(sampled_context)
        
#         if total_gen % 5000 == 0 or total_gen == num_samples:
#             logger.info(f"Generated {total_gen}/{num_samples} flows...")

#     logger.info(f"Done! Generated {total_gen} robust samples. Output saved to {OUTPUT_FILE}")

# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time  # 导入时间模块
import torch
import joblib
import logging
from sentence_transformers import SentenceTransformer

# 导入你最新包含 Cross-Attention 的模型
# 请确保 model.py 在当前目录下
from model import FlowDriveNet

# ================= 配置路径 =================
REAL_DATA_PATH = '/home/zwy/TrajFlow/ori2_48.csv'
OUTPUT_FILE = 'final_synthetic_goalflow_dataset.csv'

# 请确保这些路径指向你 train.py 生成的最新时间戳文件夹
OUTPUT_DIR = '/home/zwy/flowdrive/training_resume_20260407_111733'
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scalers_dict.pkl') 
FLOW_MODEL_PATH = os.path.join(OUTPUT_DIR, 'flow_model_epoch_300.pth')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
# ============================================

@torch.no_grad()
def generate_moderated_flow(flow_model, semantic_emb, num_features, num_steps=10, cfg_scale=2.5, sde_noise_scale=0.2):
    flow_model.eval()
    batch_size = semantic_emb.shape[0]

    # 在隐空间采样纯高斯噪声 z_0
    latent_dim = flow_model.flow.out_layer[-1].out_features
    z_t = torch.randn(batch_size, latent_dim).to(DEVICE)
    dt = 1.0 / num_steps

    uncond_emb = torch.zeros_like(semantic_emb).to(DEVICE)

    for step in range(num_steps):
        t = torch.full((batch_size,), step * dt).to(DEVICE)

        # 1. 预测速度场 (结合无分类器引导 CFG)
        if cfg_scale != 1.0:
            z_in = torch.cat([z_t, z_t], dim=0)
            t_in = torch.cat([t, t], dim=0)
            emb_in = torch.cat([uncond_emb, semantic_emb], dim=0)

            # 调用 forward_flow 在隐空间预测速度场
            v_pred = flow_model.forward_flow(z_in, t_in, emb_in)
            v_uncond, v_cond = v_pred.chunk(2, dim=0)
            v_t_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_t_pred = flow_model.forward_flow(z_t, t, semantic_emb)

        # 2. SDE 随机探索与自我纠错机制
        if step < num_steps - 1:
            noise = torch.randn_like(z_t) * np.sqrt(dt) * sde_noise_scale
        else:
            noise = 0

        # 3. 欧拉更新 + 随机微调
        z_t = z_t + v_t_pred * dt + noise

    # 生成完成后，通过解码器还原为原始物理特征
    x_generated = flow_model.decode(z_t)
    return x_generated


def main():
    logger.info(">>> 1. Loading Models and Assets...")
    if not os.path.exists(REAL_DATA_PATH):
        logger.error(f"File not found: {REAL_DATA_PATH}")
        return

    df_real = pd.read_csv(REAL_DATA_PATH).fillna(0)

    ignore_cols = ['flow_key', 'src_ip', 'dst_ip', 'src_ip_numeric', 'src_port', 'dst_port', 'proto',
                   'flowStart', 'flowEnd', 'f_flowStart', 'f_flowEnd', 'b_flowStart', 'b_flowEnd',
                   'flowEndReason', 'category', 'application_protocol', 'web_service', 'target_goal_idx']
    
    feature_cols = [col for col in df_real.columns if col not in ignore_cols]
    num_features = len(feature_cols)

    # 加载 Scaler 字典
    scalers = joblib.load(SCALER_PATH)
    
    # 加载 LLM
    llm_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

    # 初始化模型
    flow_model = FlowDriveNet(num_features=num_features, semantic_dim=384, hidden_dim=512).to(DEVICE)
    flow_model.load_state_dict(torch.load(FLOW_MODEL_PATH, map_location=DEVICE))

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    write_header = True
    total_gen = 0
    batch_size = 500
    num_samples = len(df_real)

    # --- 计时器初始化 ---
    overall_start_time = time.time()    # 总耗时计时
    interval_start_time = time.time()   # 每5000个样本的区间计时
    
    logger.info(f">>> 2. Generating High-Fidelity Diverse Traffic (Total: {num_samples})...")
    
    for start_idx in range(0, num_samples, batch_size):
        # 截取真实数据的上下文
        current_batch_size = min(batch_size, num_samples - start_idx)
        sampled_context = df_real.sample(n=current_batch_size, replace=True).copy()
        
        # 重建 Prompt
        prompts = sampled_context.apply(
            lambda row: f"A {row['category']} network flow using {row['application_protocol']} protocol "
                        f"on port {row['src_port']} to {row['dst_port']}, providing {row['web_service']} service.", 
            axis=1
        ).tolist()
        
        # 语义嵌入
        semantic_embeddings = llm_model.encode(prompts, show_progress_bar=False)
        t_semantic_emb = torch.tensor(semantic_embeddings, dtype=torch.float32).to(DEVICE)

        # 执行模型推理生成
        generated_normalized = generate_moderated_flow(
            flow_model, 
            t_semantic_emb, 
            num_features, 
            num_steps=20,
            cfg_scale=1.2, 
            sde_noise_scale=0.05
        ).cpu().numpy()

        df_gen = pd.DataFrame(columns=feature_cols)

        # 逆变换还原数据
        for i, col in enumerate(feature_cols):
            strat = scalers[col]['type']
            scaler = scalers[col]['scaler']
            col_data = generated_normalized[:, i].reshape(-1, 1)
            
            inv_data = scaler.inverse_transform(col_data).flatten()
            
            if strat in ['log_robust', 'log_standard']:
                inv_data = np.expm1(inv_data) 
                
            df_gen[col] = inv_data

        # 还原离散列
        for c in ['src_port', 'dst_port', 'proto', 'category', 'application_protocol', 'web_service']:
            df_gen[c] = sampled_context[c].values

        # 逻辑后处理
        df_gen[feature_cols] = df_gen[feature_cols].clip(lower=0)

        if 'f_pktTotalCount' in df_gen.columns:
            df_gen['pktTotalCount'] = df_gen['f_pktTotalCount'] + df_gen['b_pktTotalCount']
            df_gen['octetTotalCount'] = df_gen['f_octetTotalCount'] + df_gen['b_octetTotalCount']
        if 'max_ps' in df_gen.columns:
            df_gen['max_ps'] = df_gen[['max_ps', 'min_ps']].max(axis=1)

        for col in df_gen.columns:
            if any(k in col for k in ['Count', 'port', 'proto', 'ps']):
                if pd.api.types.is_numeric_dtype(df_gen[col]):
                    df_gen[col] = df_gen[col].round().astype(int)

        # 写入 CSV
        df_gen.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)
        write_header = False
        
        total_gen += current_batch_size
        
        # --- 核心计时逻辑：每 5000 条数据输出一次 ---
        if total_gen % 5000 == 0:
            interval_end_time = time.time()
            duration = interval_end_time - interval_start_time
            logger.info(f"[计时] 已生成 {total_gen} 条。最近 5000 条耗时: {duration:.2f} 秒 (约 {duration/60:.2f} 分钟)")
            interval_start_time = time.time() # 重置区间计时器
        
        # 常规进度显示
        if total_gen % 1000 == 0 or total_gen == num_samples:
            logger.info(f"Progress: {total_gen}/{num_samples} flows generated...")

    # 总耗时输出
    total_duration = time.time() - overall_start_time
    logger.info("-" * 50)
    logger.info(f"任务完成!")
    logger.info(f"总样本数: {total_gen}")
    logger.info(f"总总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    logger.info(f"平均速度: {total_duration/total_gen:.4f} 秒/条")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    logger.info("-" * 50)

if __name__ == "__main__":
    main()