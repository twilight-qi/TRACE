import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaLN(nn.Module):
    """Adaptive layer normalization"""

    def __init__(self, d_model, d_cond):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(d_cond, 2 * d_model)
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, condition):
        x_norm = self.norm(x)
        style = self.proj(condition).unsqueeze(1)
        scale, shift = style.chunk(2, dim=-1)
        return x_norm * (1 + scale) + shift


class SinusoidalGPSEncoder(nn.Module):
    """Sinusoidal GPS encoder"""

    def __init__(self, embed_dim, min_scale=1.0, max_scale=1000.0):
        super().__init__()
        self.num_freqs = embed_dim // 4
        self.register_buffer(
            "freqs",
            torch.logspace(
                math.log10(min_scale), math.log10(max_scale), self.num_freqs
            ),
        )

    def forward(self, coords):
        args = coords.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1) * 2 * math.pi
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1).view(
            coords.size(0), coords.size(1), -1
        )


class AdaLNTransformerEncoderLayer(nn.Module):
    """AdaLN Transformer encoder layer with padding mask support"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, d_cond):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = AdaLN(d_model, d_cond)
        self.norm2 = AdaLN(d_model, d_cond)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, mask=None, src_key_padding_mask=None, user_emb=None):
        # Pre-Norm with AdaLN
        src2 = self.norm1(src, user_emb)

        # Attention with causal and padding masks
        src2 = self.self_attn(
            src2,
            src2,
            src2,
            attn_mask=mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]

        src = src + self.dropout1(src2)

        # FFN
        src2 = self.norm2(src, user_emb)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class StandardTransformerLayer(nn.Module):
    """Standard Transformer layer with padding mask support"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, src, mask=None, src_key_padding_mask=None, user_emb=None):
        return self.layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)


class CNet(nn.Module):
    def __init__(
        self,
        num_users,
        num_locations,
        num_categories,
        embed_dim=128,
        hidden_dim=512,
        num_heads=4,
        transformer_num_layers=4,
        dropout=0.1,
        max_len=1000,
        use_history: bool = True,
        use_adaln: bool = True,
        num_bins=48,
        **kwargs,
    ):
        super().__init__()

        # Configuration
        self.use_history = use_history
        self.use_adaln = use_adaln

        self.embed_dim = hidden_dim
        self.hidden_dim = hidden_dim

        # Embeddings
        self.user_emb = nn.Embedding(num_users + 1, embed_dim)
        self.loc_emb = nn.Embedding(num_locations + 1, embed_dim)
        self.cat_emb = nn.Embedding(num_categories + 1, embed_dim)
        self.time_emb = nn.Embedding(168 + 1, embed_dim)
        self.type_emb = nn.Embedding(3, embed_dim)  # 0:Pad, 1:Hist, 2:Curr
        self.pos_emb = nn.Embedding(max_len + 1, embed_dim)
        self.gps_encoder = SinusoidalGPSEncoder(embed_dim=embed_dim)
        self.mask_token_emb = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Backbone
        if self.use_adaln:
            self.layers = nn.ModuleList(
                [
                    AdaLNTransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=num_heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=dropout,
                        d_cond=embed_dim,
                    )
                    for _ in range(transformer_num_layers)
                ]
            )
            self.final_norm = AdaLN(hidden_dim, embed_dim)
        else:
            self.layers = nn.ModuleList(
                [
                    StandardTransformerLayer(
                        d_model=hidden_dim,
                        nhead=num_heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=dropout,
                    )
                    for _ in range(transformer_num_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(hidden_dim)

        # Time Head (Distributional)
        self.max_delta = 24.0
        self.num_bins = num_bins
        self.bin_width = self.max_delta / self.num_bins

        # Register Bin Centers
        bin_centers = torch.linspace(
            self.bin_width / 2, self.max_delta - self.bin_width / 2, self.num_bins
        )
        self.register_buffer("bin_centers", bin_centers, persistent=False)

        # Prediction heads
        self.head_time_dist = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_bins),
        )

        # Category adapter
        self.cat_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.head_cat = nn.Linear(hidden_dim, num_categories + 1)
        self.head_loc = nn.Linear(hidden_dim, num_locations + 1)

    def _build_input(self, batch):
        loc, cat, slot = batch["loc"], batch["cat"], batch["time_slot"]
        type_ids, user_id, coords = batch["type_ids"], batch["user_id"], batch["coords"]

        B, L = loc.shape
        device = loc.device

        # Basic embeddings
        l = self.loc_emb(loc)
        u = self.user_emb(user_id).unsqueeze(1)  # Broadcast
        c = self.cat_emb(cat)
        t = self.time_emb(slot)

        # Position encoding
        positions = (
            torch.arange(L - 1, -1, -1, device=device).unsqueeze(0).expand(B, -1)
        )
        p = self.pos_emb(positions)

        # GPS encoding
        if coords is None:
            coords = torch.zeros(l.size(0), l.size(1), 2, device=device)
        elif coords.dim() == 2:
            coords = coords.unsqueeze(1).expand(-1, l.size(1), -1)

        g = self.gps_encoder(coords)

        # Core fusion logic
        x = l + u + t + c + g + p

        # Add type embedding if using history
        if self.use_history:
            type_feat = self.type_emb(type_ids)
            x = x + type_feat

        return x

    def get_soft_time_embedding(self, pred_slot_float):
        pred_slot_float = torch.nan_to_num(pred_slot_float, nan=0.0)
        floor_idx = torch.floor(pred_slot_float).long().clamp(0, 167)
        ceil_idx = (floor_idx + 1) % 168
        alpha = (pred_slot_float - floor_idx.float()).unsqueeze(-1)
        return (1.0 - alpha) * self.time_emb(floor_idx) + alpha * self.time_emb(
            ceil_idx
        )

    def forward(self, batch, do_masking=False, mask_ratio=0.2):
        # Input
        x_raw = self._build_input(batch)

        # Token masking during training
        if do_masking and self.training and mask_ratio > 0.0:
            B, L, D = x_raw.shape
            rand_matrix = torch.rand(B, L, device=x_raw.device)
            mask_mask = (rand_matrix < mask_ratio) & batch["mask"]
            mask_mask[:, -1] = False
            mask_token = self.mask_token_emb.expand(B, L, D)
            x_raw = torch.where(mask_mask.unsqueeze(-1), mask_token, x_raw)

        x_curr = self.input_proj(x_raw)

        # Backbone
        L = x_curr.size(1)
        device = x_curr.device

        # Build causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=device) * float("-inf"), diagonal=1
        )

        u_emb = self.user_emb(batch["user_id"])
        h_curr = x_curr

        for layer in self.layers:
            if self.use_adaln:
                h_curr = layer(h_curr, mask=causal_mask, user_emb=u_emb)
            else:
                h_curr = layer(h_curr, mask=causal_mask)

        if self.use_adaln:
            h_curr = self.final_norm(h_curr, u_emb)
        else:
            h_curr = self.final_norm(h_curr)

        # Time prediction (distributional)
        logits_time = self.head_time_dist(h_curr)  # [B, L, 48]
        probs_time = F.softmax(logits_time, dim=-1)  # [B, L, 48]

        # ArgMax for accurate predictions
        top1_idx = torch.argmax(probs_time, dim=-1)  # [B, L]
        pred_delta_argmax = self.bin_centers[top1_idx]

        # Calculate uncertainty (entropy)
        entropy = -torch.sum(
            probs_time * torch.log(probs_time + 1e-9), dim=-1, keepdim=True
        )
        uncertainty = entropy / torch.log(torch.tensor(self.num_bins))

        # Category prediction
        cat_input = h_curr
        logits_cat = self.head_cat(self.cat_adapter(cat_input))

        # Location prediction
        final_feat = h_curr
        logits_loc = self.head_loc(final_feat)

        # Return outputs
        seq_rep_cat, seq_rep_loc = None, None
        return (
            logits_loc,
            logits_cat,
            logits_time,
            pred_delta_argmax,
            uncertainty,
            seq_rep_cat,
            seq_rep_loc,
        )