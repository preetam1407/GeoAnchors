# import os
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.transforms import InterpolationMode
# from transformers import T5Tokenizer

# from dataset import MultiViewDataset

# def decode_ignore_pad(tokenizer, ids):
#     ids = ids.clone()
#     ids[ids == -100] = tokenizer.pad_token_id
#     return tokenizer.batch_decode(ids, skip_special_tokens=True)

# if __name__ == "__main__":
#     torch.set_printoptions(sci_mode=False)

#     tokenizer = T5Tokenizer.from_pretrained("t5-base")

#     img_tf = transforms.Compose([
#         transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
#         transforms.ConvertImageDtype(torch.float32),
#         transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
#                              std=(0.26862954, 0.26130258, 0.27577711)),
#     ])
#     json_path = os.path.join("data","multi_frame", "multi_frame_val.json")
#     assert os.path.isfile(json_path), f"JSON not found: {json_path}"

#     dataset = MultiViewDataset(
#         input_file=json_path,
#         tokenizer=tokenizer,
#         transform=img_tf,
#         num_views=6,
#     )
#     print(f"Dataset length: {len(dataset)}")
#     assert len(dataset) > 0, "Your JSON has 0 items."

#     # Single item check
#     with torch.inference_mode():
#         q_text, imgs, a_text, img_paths = dataset[0]
#     print("\n--- Single sample ---")
#     print("Question text:", q_text)
#     print("Answer text:", a_text)
#     print("Images tensor shape:", tuple(imgs.shape))
#     print("frame count:", imgs.shape[0])
#     print("Image paths[0..5]:")
#     for i, p in enumerate(img_paths[:6]):
#         print(f"  [{i}] {p}")
#     print("Per-image min/max (first frame):", float(imgs[0].min()), float(imgs[0].max()))

#     # DataLoader (batch_size=1 keeps RAM tiny)
#     loader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=False,
#         collate_fn=dataset.collate_fn
#     )

#     with torch.inference_mode():
#         input_ids, attention_mask, batch_imgs, labels = next(iter(loader))

#     print("\n--- Batch sample ---")
#     print("input_ids shape:", tuple(input_ids.shape))          # [1, Lq]
#     print("attention_mask shape:", tuple(attention_mask.shape))# [1, Lq]
#     print("labels shape:", tuple(labels.shape))                # [1, La]
#     print("imgs shape:", tuple(batch_imgs.shape))              # [1, 6, 3, 224, 224]

#     num_ignored = int((labels == -100).sum().item())
#     print("labels pad→-100 count:", num_ignored)

#     q_dec = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#     a_dec = decode_ignore_pad(tokenizer, labels)
#     print("\nDecoded Q[0]:", q_dec[0] if q_dec else "")
#     print("Decoded A[0]:", a_dec[0] if a_dec else "")

#     assert batch_imgs.ndim == 5 and batch_imgs.shape[1] == 6, "Expect 6 frames per sample"
#     assert input_ids.shape == attention_mask.shape, "Mask must align with input_ids"
#     print("\nAll dataset checks passed ✅")







# model.py
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat

from transformers import CLIPVisionModel, CLIPImageProcessor, T5ForConditionalGeneration

# -------------------
# Small utilities
# -------------------
def print_trainable_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

# -------------------
# AnchorFormer config
# -------------------
@dataclass
class AnchorFormerConfig:
    vision_model: str = "openai/clip-vit-large-patch14"
    feature_layer: int = -1
    tn_anchors: int = 256
    resampler_depth: int = 6
    resampler_dim_head: int = 64
    resampler_heads: int = 8
    gpa_hidden_size: int = 512
    freeze_clip: bool = True
    use_processor: bool = False

# -------------------
# Blocks
# -------------------
class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner, bias=False),
            nn.GELU(),
            nn.Linear(inner, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads

        self.norm_x = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm_x(x)
        latents = self.norm_l(latents.contiguous())
        h = self.heads

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 144,
        ff_mult: int = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        latents = getattr(self, "dynamic_query", None)
        if latents is None:
            latents = repeat(self.latents, "m d -> b m d", b=b)

        for attn, ff in self.layers:
            if self.gradient_checkpointing and self.training:
                latents = torch.utils.checkpoint.checkpoint(lambda l: attn(x, l) + l, latents, use_reentrant=False)
                latents = torch.utils.checkpoint.checkpoint(lambda l: ff(l) + l, latents, use_reentrant=False)
            else:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents
        return self.norm(latents)

# -------------------
# Anchor selection
# -------------------
@torch.no_grad()
def select_anchors_from_cls_attention(
    hidden_states: torch.Tensor,      # (B,S,D)
    attentions_last: torch.Tensor,    # (B,H,S,S)
    tn: int
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    H = attentions_last.shape[1]
    # take attention from CLS (index 0) to patches [1:]
    cls2patch = attentions_last[:, :, 0, 1:]            # (B,H,S-1)
    max_tn = min(tn, S - 1)
    per_head = max((max_tn + H - 1) // H, 1)           # ceil division

    IA = []
    for b in range(B):
        chosen = set([0])  # always keep CLS index 0
        for h in range(H):
            scores = cls2patch[b, h]                    # (S-1,)
            topk = torch.topk(scores, k=min(per_head, scores.numel())).indices + 1
            for idx in topk.tolist():
                if idx not in chosen:
                    chosen.add(idx)
                if len(chosen) - 1 >= max_tn:
                    break
            if len(chosen) - 1 >= max_tn:
                break

        if len(chosen) - 1 < max_tn:
            global_sorted = torch.argsort(cls2patch[b].mean(0), descending=True) + 1
            for idx in global_sorted.tolist():
                if idx not in chosen:
                    chosen.add(idx)
                if len(chosen) - 1 == max_tn:
                    break

        take = sorted(list(chosen))
        IA.append(hidden_states[b, take, :])
    return torch.stack(IA, dim=0)                        # (B, max_tn+1, D)

# -------------------
# GPA
# -------------------
class GPAPool(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_in))
        self.gate = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, D) -> (B, 1, D)
        B, M, D = x.shape
        q = self.query.frame(1, 1, D).expand(B, M, D)
        scores = (x * q).sum(dim=-1)                     # (B,M)
        g = torch.sigmoid(self.gate(x).squeeze(-1))      # (B,M)
        attn = torch.softmax(scores * g, dim=1)          # (B,M)
        fused = torch.einsum("bm,bmd->bd", attn, x).unsqueeze(1)  # (B,1,D)
        return fused

class AnchorFormerAggregator(nn.Module):
    def __init__(self, d_vision: int, depth: int = 6, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.resampler = PerceiverResampler(
            dim=d_vision, depth=depth, dim_head=dim_head, heads=heads,
            num_latents=1, ff_mult=4, gradient_checkpointing=False
        )

    def forward(self, Rv: torch.Tensor, IA: torch.Tensor) -> torch.Tensor:
        self.resampler.dynamic_query = IA
        return self.resampler(Rv)

class MVPAnchorFormer(nn.Module):
    def __init__(self, af_cfg: AnchorFormerConfig, lm_hidden_size: int):
        super().__init__()
        self.cfg = af_cfg
        self.clip = CLIPVisionModel.from_pretrained(af_cfg.vision_model)
        if af_cfg.freeze_clip:
            self.clip.requires_grad_(False)
        self.processor = CLIPImageProcessor.from_pretrained(af_cfg.vision_model) if af_cfg.use_processor else None
        d_vision = self.clip.config.hidden_size
        self.af = AnchorFormerAggregator(d_vision, af_cfg.resampler_depth, af_cfg.resampler_dim_head, af_cfg.resampler_heads)
        self.token_gpa = GPAPool(d_in=d_vision, d_hidden=af_cfg.gpa_hidden_size)
        self.frame_gpa = GPAPool(d_in=d_vision, d_hidden=af_cfg.gpa_hidden_size)
        self.proj = nn.Linear(d_vision, lm_hidden_size)

    def _preprocess_if_needed(self, imgs: torch.Tensor) -> torch.Tensor:
        return imgs if not self.cfg.use_processor else imgs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B,T,C,H,W) or (B,C,H,W)
        if images.ndim == 4:
            images = images.unsqueeze(1)
        B, T, C, H, W = images.shape
        device = next(self.parameters()).device

        imgs = images.reshape(B * T, C, H, W).to(device)
        imgs = self._preprocess_if_needed(imgs)

        with torch.set_grad_enabled(not self.cfg.freeze_clip):
            outs = self.clip(imgs, output_hidden_states=True, output_attentions=True)

        Rv = outs.hidden_states[self.cfg.feature_layer]   # (B*T, S, D)
        attn_last = outs.attentions[-1]                   # (B*T, H, S, S)

        IA = select_anchors_from_cls_attention(Rv, attn_last, tn=self.cfg.tn_anchors)  # (B*T, TN+1, D)
        Hv = self.af(Rv=Rv, IA=IA)                        # (B*T, 1, D) because resampler.num_latents=1
        Hv_tok = self.token_gpa(Hv)                       # (B*T, 1, D)
        Hv_tok = Hv_tok.frame(B, T, 1, -1).squeeze(2)      # (B, T, D)
        fused = self.frame_gpa(Hv_tok)                    # (B, 1, D)
        fused_proj = self.proj(fused)                     # (B, 1, lm_hidden)
        return fused_proj

# -------------------
# VLM wrapper (T5)
# -------------------
class GeoFormer(nn.Module):
    """
    Wraps MVPAnchorFormer (vision) + T5ForConditionalGeneration (LM).
    Strategy: prepend a single learned visual token (fused image) to encoder hidden states.
    """
    def __init__(self, lm_name: str = "t5-base", af_cfg: Optional[AnchorFormerConfig] = None, freeze_lm: bool = False):
        super().__init__()
        self.lm = T5ForConditionalGeneration.from_pretrained(lm_name)
        d_lm = self.lm.config.d_model
        self.vision = MVPAnchorFormer(af_cfg or AnchorFormerConfig(), lm_hidden_size=d_lm)

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                images: torch.Tensor, labels: Optional[torch.Tensor] = None):
        device = next(self.parameters()).device
        fused = self.vision(images)                # (B,1,d_lm)
        # Run encoder to get hidden states for text (without visual yet)
        enc_out = self.lm.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        enc_hid = enc_out.last_hidden_state       # (B, L, d_lm)
        # Prepend visual token
        B = enc_hid.size(0)
        enc_hid = torch.cat([fused, enc_hid], dim=1)  # (B, L+1, d_lm)
        new_mask = torch.cat([torch.ones(B, 1, device=device, dtype=attention_mask.dtype), attention_mask], dim=1)

        # Decode
        out = self.lm(encoder_outputs=(enc_hid,), attention_mask=new_mask, labels=labels, return_dict=True)
        return out

# import os, json
# root = os.path.join("data","multi_frame")
# src = os.path.join(root,"multi_frame_val.json")  # use any existing split
# with open(src) as f:
#     data = json.load(f)
# assert len(data)>0, "src JSON is empty"
# one = [data[0]]  # single QA + 6 paths

# os.makedirs(root, exist_ok=True)
# for name in ["multi_frame_train.json","multi_frame_val.json","multi_frame_test.json"]:
#     dst = os.path.join(root, name)
#     with open(dst,"w") as g: json.dump(one, g)
# print("Wrote 1-sample train/val/test JSONs under", root)