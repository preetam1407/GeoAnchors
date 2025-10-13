import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn 
from torch import einsum
from einops import rearrange, repeat

from transformers import CLIPVisionModel, CLIPImageProcessor, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput  # add near your imports

def _dbg_once(obj, key: str, msg: str):
    # prints only once per unique key on this object
    if not getattr(obj, f"__dbg_{key}", False):
        print(msg)
        setattr(obj, f"__dbg_{key}", True)

DEBUG = os.getenv("VQA_DEBUG", "1") == "1" 

def print_trainable_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


@dataclass
class AnchorFormerConfig:
    vision_model: str = "openai/clip-vit-large-patch14"
    feature_layer: int = -1
    tn_anchors: int = 256
    resampler_depth: int = 6
    resampler_dim_head: int = 64
    resampler_heads: int = 8
    gpa_hidden_size: int  = 512
    freeze_clip: bool  = True
    use_processor: bool = False

#mlp inside transformer block
class FeedForward(nn.Module):
    """Pre-LN MLP: LN->Linear(D,4D) ->GELU ->Linear(4D,D)"""
    def __init__(self, dim:int, mult: int = 4):
        super().__init__()
        inner = int(dim*mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner, bias = False),
            nn.GELU(),
            nn.Linear(inner, dim, bias = False),           
        )

    def forward(self, x:torch.tensor) ->torch.tensor:
        return self.net(x)
    
#queries = anchors (learned) and key/value are all vision tokens
class PerceiverAttention(nn.Module):
    """
    Cross-attention: queries(latents) attend the media tokens (x)
    Input:
    x: 
    (B,S,D)   #Key/values (vision toeksn; include cls)
    latents: 
    (B,M,D)  #queries (IA = CLS + TN selected patches)
    Output:
    (B,M,D)    #Updated queries 
    """
    def __init__(self, dim:int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner = dim_head *heads
        
        #pre-norm for k/v from media tokens and q from latents
        self.norm_x = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        
        #Linear projection: 
        self.to_q = nn.Linear(dim, inner, bias = False)
        self.to_kv = nn.Linear(dim, inner*2, bias = False)   #produce k|v and then chunk.
        self.to_out = nn.Linear(inner, dim, bias = False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm_x(x)
        latents = self.norm_l(latents.contiguous())
        h = self.heads

        q = self.to_q(latents)
        k,v = self.to_kv(x).chunk(2, dim=-1)

        q,k,v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q,k,v))
        q = q*self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim - sim.amax(dim = -1, keepdim  = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        if DEBUG:
            _dbg_once(self, "pa_shapes",
                f"[PerceiverAttention] qkv-> out {tuple(out.shape)} | heads={self.heads}")
        return self.to_out(out)
    
# stack of corss-attn and ffn with residuals
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

        #Learnable queries if dynamic absent, used by google deepmind
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, mean = 0.0, std = 0.02)

        #Build 'depth layers, cross->ffn ...
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)    #addedddddddddddddd
        
    def forward(self, x:torch.Tensor) ->torch.Tensor:
        """
        x: (B,S,D)  #include cls in key/values
        return: (B,M,D)
        """
        b = x.shape[0]
        latents = getattr(self, "dynamic_query", None)
        if latents is None:
            latents =  repeat(self.latents, "m d -> b m d", b=b)

        for attn, ff in self.layers:
            if self.gradient_checkpointing and self.training:
                latents = torch.utils.checkpoint.checkpoint(
                    lambda l: attn(x,l)+l, latents, use_reentrant = False
                )
                latents = torch.utils.checkpoint.checkpoint(
                    lambda l: ff(l) + l, latents, use_reentrant = False
                )
            else:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents

        if DEBUG:
            _dbg_once(self, "resampler_latents",
                f"[Resampler] latents {tuple(latents.shape)} | layers={len(self.layers)} | ckpt={self.gradient_checkpointing}")

        return self.norm(latents)
    
#Anchor selection
@torch.no_grad()
def select_anchors_from_cls_attention(
        hidden_states: torch.Tensor,
        attentions_last: torch.Tensor,
        tn: int
) -> torch.Tensor:
    B,S,D = hidden_states.shape
    H = attentions_last.shape[1]
    cls2patch = attentions_last[:,:,0,1:]
    max_tn = min(tn, S-1)

    per_head = max((max_tn + H - 1) // H, 1)           # ceil division
    IA = []

    for b in range(B):
        chosen = set([0])
        for h in range(H):
            scores = cls2patch[b, h]                    
            topk = torch.topk(scores, k=min(per_head, scores.numel())).indices + 1
            for idx in topk.tolist():
                if idx not in chosen:
                    chosen.add(idx)
                if len(chosen) - 1 >= max_tn:
                    break
            if len(chosen) - 1 >= max_tn:
                break

        # if still short (global fill)
        if len(chosen) -1 < max_tn:
            global_sorted = (torch.argsort(cls2patch[b].mean(0))+1).tolist()[::-1]
            for idx in global_sorted:
                if idx not in chosen:
                    chosen.add(int(idx))
                    if len(chosen) -1 == max_tn:
                        break

        take = sorted(list(chosen))
        IA.append(hidden_states[b,take, :])

    IA = torch.stack(IA, dim = 0)
    return IA

# GPA Pooling of 6 images
class GPAPool(nn.Module):
    """
    Gated pooling attention over a set of token:(B,M,D)->(B,1,D)
    """
    def __init__(self, d_in:int, d_hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_in)) #Learnable global query vector
        self.gate = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        B,M,D = x.shape
        q = self.query.view(1,1,D).expand(B,M,D)
        scores = (x*q).sum(dim=-1)
        g =  torch.sigmoid(self.gate(x).squeeze(-1))
        attn = torch.softmax(scores*g, dim=1)
        fused = torch.einsum('bm,bmd->bd', attn, x).unsqueeze(1)

        if DEBUG:
            with torch.no_grad():
                g_min, g_max, g_mean = float(g.min().item()), float(g.max().item()), float(g.mean().item())
            _dbg_once(self, "gpa_gate", f"[GPA] gate min={g_min:.3f} max={g_max:.3f} mean={g_mean:.3f}")

        return fused
    
class AnchorFormerAggregator(nn.Module):
    """
    wrapper: set resampler.dynamic_query = IA, then resample with keys/values R_v
    """
    def __init__(self, d_vision: int, depth:int = 6, dim_head: int=64, heads:int =8):
        super().__init__()
        self.resampler = PerceiverResampler(
            dim = d_vision,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=1,
            ff_mult=4,
            gradient_checkpointing=False
        )
    
    def forward(self, Rv: torch.Tensor, IA:torch.Tensor)->torch.Tensor:
        self.resampler.dynamic_query = IA
        return self.resampler(Rv)

class MVPAnchorFormer(nn.Module):
    """
    End-to-end per-image AnchorFormer + multi-view GPA + projection to LM width.

    Forward:
      images: (B, T, C, H, W) or (B, C, H, W)
      returns: (B, 1, lm_hidden_size)
    """
    def __init__(self, af_cfg: AnchorFormerConfig, lm_hidden_size: int):
        super().__init__()                               # Init module
        self.cfg = af_cfg                                # Save config

        # ---- Vision tower (CLIP) ----
        self.clip = CLIPVisionModel.from_pretrained(af_cfg.vision_model, attn_implementation="eager")
        if af_cfg.freeze_clip:
            self.clip.requires_grad_(False)              # Freeze weights if requested

        self.processor = None
        if af_cfg.use_processor:
            self.processor = CLIPImageProcessor.from_pretrained(af_cfg.vision_model)  

        d_vision = self.clip.config.hidden_size          

        self.af = AnchorFormerAggregator(
            d_vision=d_vision,
            depth=af_cfg.resampler_depth,
            dim_head=af_cfg.resampler_dim_head,
            heads=af_cfg.resampler_heads,
        )

        self.token_gpa = GPAPool(d_in=d_vision, d_hidden=af_cfg.gpa_hidden_size)  
        self.frame_gpa = GPAPool(d_in=d_vision, d_hidden=af_cfg.gpa_hidden_size)  

        self.proj = nn.Linear(d_vision, lm_hidden_size)

    def _preprocess_if_needed(self, imgs: torch.Tensor) -> torch.Tensor:
        return imgs if not self.cfg.use_processor else imgs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, T, C, H, W) or (B, C, H, W)
        return: (B, 1, lm_hidden_size)
        """
        if images.ndim == 4:
            images = images.unsqueeze(1)                 # If single frame -> add T=1 dimension
        assert images.ndim == 5, "images must be (B, T, C, H, W) or (B, C, H, W)"

        B, T, C, H, W = images.shape                     # Unpack shapes
        device = next(self.parameters()).device          # Current module device (cpu/cuda)

        imgs = images.reshape(B * T, C, H, W).to(device) # Flatten frames into batch (B*T, C, H, W)

        imgs = self._preprocess_if_needed(imgs)          # Optional in-module preprocessing

        # ---- CLIP forward pass ----
        with torch.set_grad_enabled(not self.cfg.freeze_clip):  # Save grads if CLIP is trainable
            outs = self.clip(
                imgs,
                output_hidden_states=True,               # We need tokens from a chosen layer
                output_attentions=True                   # We need last-layer attentions for selection
            )

        # Tokens (include CLS) from the chosen feature layer (often last = -1)
        Rv = outs.hidden_states[self.cfg.feature_layer]  # Shape: (B*T, S, D)

        # Last-layer attentions (paper uses this for anchor selection)
        attn_last = outs.attentions[-1]                  # Shape: (B*T, H, S, S)

        # ---- Anchor selection (CLS + TN top patches per image) ----
        IA = select_anchors_from_cls_attention(
            Rv, attn_last, tn=self.cfg.tn_anchors
        )              
        if DEBUG:
            BxT, S, D = Rv.shape
            H = attn_last.shape[1]
            _dbg_once(self, "clip_tokens",
                f"[CLIP] Rv {tuple(Rv.shape)} | attn_last {tuple(attn_last.shape)} | H={H}")
            _dbg_once(self, "anchors",
                f"[Anchors] IA {tuple(IA.shape)} | requested_tn={self.cfg.tn_anchors} | cap={min(self.cfg.tn_anchors, S-1)}")
                                        # Shape: (B*T, TN+1, D)

        # ---- Cross-attention aggregation (Perceiver) ----
        Hv = self.af(Rv=Rv, IA=IA)                      # Refined anchors: (B*T, TN+1, D)
        if DEBUG:
            with torch.no_grad():
                hv_isfinite = torch.isfinite(Hv).all().item()
            _dbg_once(self, "hv_stats",
                f"[AF] Hv {tuple(Hv.shape)} | finite={hv_isfinite}")


        # ---- GPA pooling strategy ----
        # 1) Pool within each frame's anchors -> (B*T, 1, D)
        Hv_pooled = self.token_gpa(Hv)                  # (B*T, 1, D)

        if DEBUG:
            with torch.no_grad():
                token_min = float(Hv_pooled.min().item())
                token_max = float(Hv_pooled.max().item())
            _dbg_once(self, "token_gpa",
                f"[GPA-token] {tuple(Hv_pooled.shape)} | min={token_min:.3f} max={token_max:.3f}")


        # Reshape back to (B, T, D) by removing the single-token dim
        Hv_pooled = Hv_pooled.view(B, T, 1, -1).squeeze(2)  # (B, T, D)

        # 2) Pool across frames -> (B, 1, D)
        fused = self.frame_gpa(Hv_pooled)               # (B, 1, D)

        if DEBUG:
            with torch.no_grad():
                frame_min = float(fused.min().item())
                frame_max = float(fused.max().item())
            _dbg_once(self, "frame_gpa",
                f"[GPA-frame] {tuple(fused.shape)} | min={frame_min:.3f} max={frame_max:.3f}")


        # ---- Final projection to LM space ----
        fused_proj = self.proj(fused)                   # (B, 1, lm_hidden_size)
        if DEBUG:
            _dbg_once(self, "proj",
                f"[Proj] fused_proj {tuple(fused_proj.shape)} -> lm_hidden={self.proj.out_features}")
        return fused_proj



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
        if DEBUG:
            _dbg_once(self, "enc_shape",
                f"[LM-enc] enc_hid {tuple(enc_hid.shape)} | text_len={attention_mask.size(1)}")

        # Prepend visual token
        B = enc_hid.size(0)
        enc_hid = torch.cat([fused, enc_hid], dim=1)  # (B, L+1, d_lm)
        new_mask = torch.cat([torch.ones(B, 1, device=device, dtype=attention_mask.dtype), attention_mask], dim=1)

        if DEBUG:
            with torch.no_grad():
                mask_added = int(new_mask[:, 0].sum().item())
                total_len = new_mask.size(1)
            _dbg_once(self, "mask_check",
                f"[LM-mask] new_mask {tuple(new_mask.shape)} | first_col_sum={mask_added} | total_len={total_len}")


        # Decode
        out = self.lm(encoder_outputs=(enc_hid,), attention_mask=new_mask, labels=labels, return_dict=True)

                
        if DEBUG:
            _dbg_once(self, "lm_logits",
                f"[LM-dec] logits {tuple(out.logits.shape)} | loss_present={out.loss is not None}")
        return out
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 images: torch.Tensor, **gen_kwargs):
        """
        Runs vision + encoder, then calls T5.generate() with the fused encoder outputs.
        """
        device = next(self.parameters()).device
        self.eval()

        # fused visual token
        fused = self.vision(images)  # (B,1,d_lm)

        # text encoder
        enc_out = self.lm.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        enc_hid = enc_out.last_hidden_state  # (B, L, d_lm)

        # prepend visual token + mask
        B = enc_hid.size(0)
        enc_hid = torch.cat([fused, enc_hid], dim=1)  # (B, L+1, d_lm)
        new_mask = torch.cat(
            [torch.ones(B, 1, device=device, dtype=attention_mask.dtype), attention_mask], dim=1
        )

        enc_outputs = BaseModelOutput(last_hidden_state=enc_hid)
        # sensible defaults; override with gen_kwargs if you want
        gen_defaults = dict(max_new_tokens=16, num_beams=1, do_sample=False)
        gen_defaults.update(gen_kwargs)

        return self.lm.generate(
            encoder_outputs=enc_outputs,
            attention_mask=new_mask,
            **gen_defaults
        )
