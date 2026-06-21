import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ConvNextV2Model
from .layers import AttentionMIL, CausalDisentangler
from typing import Dict


class CausalConvNeXtMIL(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/convnextv2-tiny-1k-224",
        num_classes: int = 2,
        mil_dropout: float = 0.5,
        # PIC-MIL++ switches
        use_pce: bool = False,
        use_pscg: bool = False,
        use_cbn: bool = False,
        # Strength of PSCG / CBN interventions (0~1)
        lambda_pscg: float = 0.3,
        lambda_cbn: float = 0.5,
        # Attention temperature
        attn_temp: float = 2.0,
    ):
        super().__init__()

        # Switches for PIC-MIL++
        self.use_pce = use_pce
        self.use_pscg = use_pscg
        self.use_cbn = use_cbn
        self.lambda_pscg = lambda_pscg
        self.lambda_cbn = lambda_cbn

        # 1. RCM Backbone
        self.rcm_encoder = ConvNextV2Model.from_pretrained(model_name)
        feat_dim = self.rcm_encoder.config.hidden_sizes[-1]

        # 2. Path Backbone (Teacher)
        self.path_encoder = ConvNextV2Model.from_pretrained(model_name)
        for param in self.path_encoder.parameters():
            param.requires_grad = False

        # 3. Disentangler
        self.content_dim = 512
        self.disentangler = CausalDisentangler(
            in_dim=feat_dim,
            content_dim=self.content_dim,
            style_dim=256,
        )

        # 4. Projector (Path -> content space)
        self.path_proj = nn.Sequential(
            nn.Linear(feat_dim, self.content_dim),
            nn.ReLU(),
            nn.LayerNorm(self.content_dim),
        )


        self.mil = AttentionMIL(
            self.content_dim,
            dropout=mil_dropout,
            num_classes=num_classes,
            attn_temp=attn_temp,
        )


        # Patient Context Encoder (PCE): mean + LayerNorm + detach
        self.pce_ln = nn.LayerNorm(self.content_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.content_dim * 2, self.content_dim),
            nn.ReLU(),
            nn.Linear(self.content_dim, 1),
        )


        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.path_proj, self.mil, self.disentangler]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, encoder, images: torch.Tensor) -> torch.Tensor:
        out = encoder(pixel_values=images, output_hidden_states=False, return_dict=True)
        return out.last_hidden_state.mean(dim=(2, 3))

    def _apply_pic_mil(self, patient_feats: torch.Tensor):
        feats_for_attn = patient_feats
        gate = None
        alpha = None


        if not (self.use_pce or self.use_pscg or self.use_cbn):
            return feats_for_attn, gate, alpha

        # ----- Patient Context Encoder (PCE) -----

        z_patient = patient_feats.mean(dim=0, keepdim=True)  # [1, D]
        z_patient = self.pce_ln(z_patient)
        z_patient = z_patient.detach()

        # ----- Patient-Aware Soft Causal Gate (PSCG) -----
        if self.use_pscg:

            z_expand = z_patient.expand_as(patient_feats)  # [N, D]
            gate_in = torch.cat([patient_feats, z_expand], dim=1)  # [N, 2D]
            gate = torch.sigmoid(self.gate_mlp(gate_in))


            h_patient = patient_feats * gate
            h_gate = patient_feats * (1.0 - gate)
            h_causal = (1.0 - self.lambda_pscg) * patient_feats + self.lambda_pscg * h_gate
        else:
            h_causal = patient_feats

        # ----- Causal Budget Normalization (CBN) -----
        if self.use_cbn:
            # Step 1: a_i = ||h_causal_i||_2
            a = h_causal.norm(p=2, dim=1)  # [N]

            A = a.sum() + 1e-6
            N = a.size(0)
            alpha = (a / A * float(N)).unsqueeze(1)  # [N, 1]

            h_cbn = h_causal * alpha
            h_causal = (1.0 - self.lambda_cbn) * h_causal + self.lambda_cbn * h_cbn

        feats_for_attn = h_causal
        return feats_for_attn, gate, alpha

    def forward(self, rcm_images, bag_sizes, path_images=None):
        results: Dict[str, torch.Tensor] = {}

        # === 1. RCM Stream ===
        rcm_raw = self.extract_features(self.rcm_encoder, rcm_images)

        # Disentangle: Z -> (Content, Style)
        c_rcm, s_rcm = self.disentangler(rcm_raw)

        # Self-Reconstruction: (C, S) -> Z_hat
        rec_rcm = self.disentangler.reconstruct(c_rcm, s_rcm)

        results.update({
            "rcm_raw": rcm_raw,
            "c_rcm": c_rcm,
            "s_rcm": s_rcm,
            "rec_rcm": rec_rcm,
            "c_prime": None,
        })

        # === [NEW] Style Swapping Intervention (Training Only) ===
        if self.training:
            # 1. Shuffle Style within the batch
            perm = torch.randperm(s_rcm.size(0), device=s_rcm.device)
            s_shuffled = s_rcm[perm]

            # 2. Hybrid Reconstruction: Content(A) + Style(B) -> Feature(Hybrid)
            z_hybrid = self.disentangler.reconstruct(c_rcm, s_shuffled)

            # 3. Cycle Consistency Check
            c_prime, _ = self.disentangler(z_hybrid)
            results["c_prime"] = c_prime

        # === 2. MIL Aggregation (c_rcm -> [PIC-MIL++] -> Attention-MIL) ===
        splits = torch.split(c_rcm, bag_sizes, dim=0)
        bag_logits_list = []
        bag_feats_list = []


        gate_list = []
        alpha_list = []
        attn_list = []

        for patient_feats in splits:
            # patient_feats: [N, D] for a single patient (N patches)
            feats_for_attn, gate, alpha = self._apply_pic_mil(patient_feats)


            logits, bag_feat, attn = self.mil(feats_for_attn)
            bag_logits_list.append(logits)
            bag_feats_list.append(bag_feat)

            if gate is not None:
                gate_list.append(gate.detach().cpu())
            if alpha is not None:
                alpha_list.append(alpha.detach().cpu())
            attn_list.append(attn.detach().cpu())

        results["logits"] = torch.cat(bag_logits_list, dim=0)
        results["patient_content"] = torch.cat(bag_feats_list, dim=0)


        if gate_list:
            g = torch.cat(gate_list, dim=0)  # [sum_N, 1]
            results["gate_mean"] = g.mean()
            results["gate_min"] = g.min()
            results["gate_max"] = g.max()
        if alpha_list:
            a_all = torch.cat(alpha_list, dim=0)  # [sum_N, 1]
            results["alpha_mean"] = a_all.mean()
            results["alpha_min"] = a_all.min()
            results["alpha_max"] = a_all.max()
        if attn_list:
            w_all = torch.cat(attn_list, dim=0)  # [sum_N, 1]
            results["attn_mean"] = w_all.mean()
            results["attn_min"] = w_all.min()
            results["attn_max"] = w_all.max()

        # === 3. Path Stream ===
        if path_images is not None:
            with torch.no_grad():
                path_raw = self.extract_features(self.path_encoder, path_images)
            c_path = self.path_proj(path_raw)
            results["c_path"] = c_path
            path_logits = self.mil.cls(self.mil.dropout(c_path))
            results["cf_logits"] = path_logits

        return results

