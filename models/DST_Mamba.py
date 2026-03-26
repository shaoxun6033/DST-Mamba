import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    print(
        "Warning: 'mamba_ssm' not found. Please install via 'pip install mamba-ssm'. Using a Mock Linear layer for testing.")
    Mamba = None


class MambaWrapper(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is not None:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba = nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.GELU(),
                nn.Linear(d_model * expand, d_model)
            )

    def forward(self, x):
        return self.mamba(x)


# ==========================================
# 1. RevIN (Reversible Instance Normalization)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# ==========================================
# 2. Residual Learnable Lifting Wavelet
# ==========================================
class Predictor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        in_channels = configs.enc_in
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )

    def forward(self, x_e):
        return x_e + self.net(x_e)


class Updater(nn.Module):
    def __init__(self, configs):
        super().__init__()
        in_channels = configs.enc_in
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )

    def forward(self, d):
        return self.net(d)


class LiftingWaveletBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.predictor = Predictor(configs)
        self.updater = Updater(configs)

    def decompose(self, x):
        x_e = x[:, :, 0::2]
        x_o = x[:, :, 1::2]
        d = x_o - self.predictor(x_e)
        c = x_e + self.updater(d)
        return c, d

    def reconstruct(self, c, d):
        x_e = c - self.updater(d)
        x_o = d + self.predictor(x_e)

        B, C, L_half = x_e.shape
        x = torch.zeros(B, C, L_half * 2, device=x_e.device)
        x[:, :, 0::2] = x_e
        x[:, :, 1::2] = x_o
        return x


# ==========================================
# class GraphAdjGenerator(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         d_model = configs.d_model
#         self.query = nn.Linear(d_model, d_model // 2)
#         self.key = nn.Linear(d_model, d_model // 2)
#         self.temperature = (d_model // 2) ** 0.5

#     def forward(self, x):
#         x_pooled = x.mean(dim=1) # (B, C, D)
#         q = self.query(x_pooled)
#         k = self.key(x_pooled)
#         attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
#         P = F.softmax(attn, dim=-1) # (B, C, C)
#         return P

class GraphAdjGenerator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        d_model = configs.d_model
        self.node_proj = nn.Linear(d_model, d_model // 2)
        self.temperature = (d_model // 2) ** 0.5

    def forward(self, x):
        x_pooled = x.mean(dim=1)  # (B, C, D)
        node_emb = self.node_proj(x_pooled)  # (B, C, D/2)

        attn = torch.matmul(node_emb, node_emb.transpose(-2, -1)) / self.temperature

        P = torch.sigmoid(attn)  # (B, C, C)
        return P


class DualMambaBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        d_model = configs.d_model

        self.temporal_mamba = MambaWrapper(d_model=d_model)
        self.graph_generator = GraphAdjGenerator(configs)
        self.channel_mamba = MambaWrapper(d_model=d_model)

    def forward(self, x):
        B, L, C, D = x.shape

        x_t = x.transpose(1, 2).reshape(B * C, L, D)
        T_out = self.temporal_mamba(x_t)
        T_out = T_out.reshape(B, C, L, D).transpose(1, 2)

        P = self.graph_generator(x)
        C_permuted = torch.einsum('bcc, blcd -> blcd', P, x)

        x_c = C_permuted.reshape(B * L, C, D)
        C_out = self.channel_mamba(x_c)
        C_out = C_out.reshape(B, L, C, D)

        return T_out, C_out


class FiLMFusion(nn.Module):
    def __init__(self, configs):
        super().__init__()
        d_model = configs.d_model

        self.film = nn.Linear(d_model, d_model * 2)
        self.gate = nn.Linear(d_model * 2, 1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, T, C):
        B, L, Ch, D = T.shape

        T_flat = T.reshape(B, L * Ch, D)
        C_flat = C.reshape(B, L * Ch, D)

        gamma_beta = self.film(C_flat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        modulated = gamma * T_flat + beta

        g = torch.sigmoid(self.gate(torch.cat([T_flat, C_flat], dim=-1)))

        H_flat = T_flat + g * self.ffn(modulated)
        H_flat = self.norm(H_flat)

        return H_flat.reshape(B, L, Ch, D)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.revin = RevIN(num_features=self.enc_in)
        self.embedding = nn.Linear(1, self.d_model)

        self.wavelet_level1 = LiftingWaveletBlock(configs)
        self.wavelet_level2 = LiftingWaveletBlock(configs)

        self.mamba_c2 = DualMambaBlock(configs)
        self.mamba_d2 = DualMambaBlock(configs)
        self.mamba_d1 = DualMambaBlock(configs)

        self.fusion_c2 = FiLMFusion(configs)
        self.fusion_d2 = FiLMFusion(configs)
        self.fusion_d1 = FiLMFusion(configs)

        self.out_proj = nn.Linear(self.d_model, 1)
        self.pred_head = nn.Linear(self.seq_len, self.pred_len)

    def _process_component(self, comp, mamba_block, fusion_block):
        comp_emb = comp.transpose(1, 2).unsqueeze(-1)
        comp_emb = self.embedding(comp_emb)

        T, C_feat = mamba_block(comp_emb)

        H = fusion_block(T, C_feat)

        H_out = self.out_proj(H).squeeze(-1).transpose(1, 2)
        return H_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, adj=None):
        x_norm = self.revin(x_enc, mode='norm')
        x_w = x_norm.transpose(1, 2)

        c1, d1 = self.wavelet_level1.decompose(x_w)
        c2, d2 = self.wavelet_level2.decompose(c1)

        c2_out = self._process_component(c2, self.mamba_c2, self.fusion_c2)
        d2_out = self._process_component(d2, self.mamba_d2, self.fusion_d2)
        d1_out = self._process_component(d1, self.mamba_d1, self.fusion_d1)

        c1_recon = self.wavelet_level2.reconstruct(c2_out, d2_out)
        x_recon = self.wavelet_level1.reconstruct(c1_recon, d1_out)

        x_recon = x_recon.transpose(1, 2)
        out = self.pred_head(x_recon.transpose(1, 2)).transpose(1, 2)

        final_out = self.revin(out, mode='denorm')
        return final_out