# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F
import ot  

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"model_training_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("LatentFlow")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AdaptiveCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, time_cond_dim, semantic_dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Cross-Attention: 注入 LLM 语义特征
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        # 自适应门控，决定特征吸纳多少语义
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_cond_dim, 3 * hidden_dim)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t_cond, semantic_emb):
        # 时间条件注入
        shift, scale, gate = self.adaLN_modulation(t_cond).unsqueeze(1).chunk(3, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift

        # 语义注入
        k_v = self.semantic_proj(semantic_emb).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=h, key=k_v, value=k_v)

        # 自适应门控过滤
        context_gate = self.adaptive_gate(h)
        h = h + context_gate * attn_out

        return x + gate * self.mlp(self.norm2(h))


class TabularAutoEncoder(nn.Module):
    def __init__(self, num_features, latent_dim=64, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class LatentFlowNet(nn.Module):
    def __init__(self, latent_dim, semantic_dim, time_dim, hidden_dim):
        super().__init__()
        self.time_mlp = SinusoidalPositionEmbeddings(time_dim)
        self.in_layer = nn.Linear(latent_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            AdaptiveCrossAttentionBlock(hidden_dim, time_dim, semantic_dim, num_heads=8) 
            for _ in range(6)
        ])
        
        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, t, semantic_emb):
        t_emb = self.time_mlp(t)
        h = self.in_layer(z_t).unsqueeze(1)
        for block in self.blocks:
            h = block(h, t_emb, semantic_emb)
        return self.out_layer(h.squeeze(1))

class FlowDriveNet(nn.Module):
    def __init__(self, num_features, semantic_dim=384, time_dim=128, latent_dim=64, hidden_dim=768):
        super().__init__()
        self.ae = TabularAutoEncoder(num_features, latent_dim=latent_dim, hidden_dim=hidden_dim//2)
        self.flow = LatentFlowNet(latent_dim, semantic_dim, time_dim, hidden_dim)

    def forward(self, x_t, t, semantic_emb):
        return self.flow(x_t, t, semantic_emb)

    def encode(self, x):
        return self.ae.encoder(x)

    def decode(self, z):
        return self.ae.decoder(z)

    def forward_flow(self, z_t, t, semantic_emb):
        return self.flow(z_t, t, semantic_emb)



def train_flowdrive(dataloader, model, epochs=1800, device='cuda', save_dir=None, save_interval=50, start_epoch=0, ae_epochs=100):
    model.to(device)
    

    ae_optimizer = torch.optim.AdamW(model.ae.parameters(), lr=1e-3, weight_decay=1e-4)
    flow_optimizer = torch.optim.AdamW(model.flow.parameters(), lr=5e-4, weight_decay=1e-4)
    
    for epoch in range(start_epoch, epochs):
        epoch_ae_loss = 0.0
        epoch_flow_loss = 0.0
        
        is_ae_phase = epoch < ae_epochs
        if is_ae_phase:
            model.ae.train()
            model.flow.eval()
        else:
            model.ae.eval()
            model.flow.train()

        for batch in dataloader:
            x_real = batch['continuous_features'].to(device)
            semantic_emb = batch['semantic_emb'].to(device)
            sample_weights = batch['weight'].to(device)
            batch_size = x_real.shape[0]

            if is_ae_phase:
                ae_optimizer.zero_grad()
                x_recon, _ = model.ae(x_real)
                
                mse_unreduced = F.mse_loss(x_recon, x_real, reduction='none').mean(dim=1)
                loss_ae = (mse_unreduced * sample_weights).mean()
                
                loss_ae.backward()
                ae_optimizer.step()
                epoch_ae_loss += loss_ae.item()
            else:
                with torch.no_grad():
                    z_1 = model.encode(x_real) # 获取真实数据的隐变量

                z_0_random = torch.randn_like(z_1).to(device)
                
                cost_matrix = torch.cdist(z_0_random, z_1).pow(2).cpu().numpy()
                
                a = np.ones(batch_size) / batch_size
                b = np.ones(batch_size) / batch_size
                
                pi = ot.emd(a, b, cost_matrix)
                
                assignment = np.argmax(pi, axis=1)
                z_0 = z_0_random[assignment]
                # ----------------------------------------------------

                t = torch.rand(batch_size).to(device)
                t_expand = t.view(-1, 1)
                z_t = t_expand * z_1 + (1 - t_expand) * z_0
                v_target = z_1 - z_0

                flow_optimizer.zero_grad()
                v_pred = model.forward_flow(z_t, t, semantic_emb)
                
                loss_unreduced = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=1)
                loss_flow = (loss_unreduced * sample_weights).mean()
                
                loss_flow.backward()
                flow_optimizer.step()
                epoch_flow_loss += loss_flow.item()

        if is_ae_phase:
            avg_loss = epoch_ae_loss / len(dataloader)
            logger.info(f"Epoch [{epoch + 1}/{epochs}] [Phase 1: AE Pretrain] Recon Loss: {avg_loss:.6f}")
        else:
            avg_loss = epoch_flow_loss / len(dataloader)
            logger.info(f"Epoch [{epoch + 1}/{epochs}] [Phase 2: Latent Flow + OT] Vector Field Loss: {avg_loss:.6f}")

        if save_dir is not None and (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f'flow_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"--> Checkpoint saved: {ckpt_path}")

    return model
