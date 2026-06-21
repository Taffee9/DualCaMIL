import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        attn_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
        attn_temp: float = 1.0,
    ):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.attn_temp = attn_temp
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [N, C]
        attn_logits = self.attn(feats)
        if self.attn_temp != 1.0:
            attn_logits = attn_logits / self.attn_temp
        attn_weights = torch.softmax(attn_logits, dim=0)

        bag_feat = torch.sum(attn_weights * feats, dim=0, keepdim=True)  # [1, C]
        bag_feat = self.dropout(bag_feat)
        logits = self.cls(bag_feat)
        return logits, bag_feat, attn_weights


class CausalDisentangler(nn.Module):
    def __init__(self, in_dim=768, content_dim=512, style_dim=256):
        super().__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim

        # Encoder: Z -> S
        self.content_enc = nn.Sequential(
            nn.Linear(in_dim, content_dim),
            nn.LayerNorm(content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, content_dim)
        )

        # Encoder: Z -> N
        self.style_enc = nn.Sequential(
            nn.Linear(in_dim, style_dim),
            nn.LayerNorm(style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )


        self.decoder = nn.Sequential(
            nn.Linear(content_dim + style_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

    def forward(self, x):
        c = self.content_enc(x)
        s = self.style_enc(x)
        return c, s

    def reconstruct(self, c, s):
        return self.decoder(torch.cat([c, s], dim=1))


class OrthogonalDeconfoundMIL(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128, dropout=0.2, num_classes=2):
        super().__init__()


        self.style_attn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )


        self.causal_attn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )


        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        prior_pos = 0.6
        logit_bias = torch.log(torch.tensor(prior_pos / (1 - prior_pos)))
        with torch.no_grad():
            self.classifier[-1].bias[:] = torch.tensor([0.0, logit_bias])

    def forward(self, bag_features):

        style_scores = self.style_attn(bag_features)  # [N, 1]
        style_weights = F.softmax(style_scores, dim=0)
        z_style = torch.sum(bag_features * style_weights, dim=0, keepdim=True)  # [1, D]


        dot_prod = torch.mm(bag_features, z_style.t())  # [N, 1]
        style_norm_sq = torch.sum(z_style * z_style) + 1e-6
        h_proj = (dot_prod / style_norm_sq) * z_style
        h_causal = bag_features - h_proj
        h_causal = h_causal + 0.1 * bag_features

        mean_norm_feat = bag_features.norm(dim=1).mean()
        mean_norm_causal = h_causal.norm(dim=1).mean() + 1e-6
        scale = (mean_norm_feat / mean_norm_causal).clamp(min=0.1, max=5.0)
        h_causal = h_causal * scale


        causal_scores = self.causal_attn(h_causal)
        causal_weights = F.softmax(causal_scores, dim=0)
        z_bag_final = torch.sum(h_causal * causal_weights, dim=0, keepdim=True)  # [1, D]

        logits = self.classifier(z_bag_final)
        return logits, z_style.squeeze(0), z_bag_final.squeeze(0)

