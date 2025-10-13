import math
import dataclasses as dataclass
from typing import Optional
import torch
import torch.nn as nn 
from torch import einsum

from einops import rearrange, repeat
from transformers import CLIPVisionModel, CLIPImageProcessor

@dataclass
class AnchorFromerConfig:
    vision_model: str = "openai/clip-vit-large-patch14"
    feature_layer: int = -1
    tn_anchors: int = 256
    resampler_depth: int = 6
    resampler_dim_head: int = 8
    resampler_heads: int = 512
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

    def forward(Self, x:torch.tensor) ->torch.tensor:
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

        out = einsum("b h i d, b h j d -> b h i j", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)
    
# stack of corss-attn and ffn with residuals
class PereceiverResampler(nn.Module):
    def _init__(
            self,
            dim: int,
            depth: int = 6,
            dim_head: int = 64,
            heads: int = 8,
            num_latents: int = 144,
            ff_mult: int = 4,
            gradient_checkpointing: book = False,
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
        
    def forward(self, x:torch.Tensor) ->torch.Tensor:
        """
        x: (B,S,D)  #include cls in key/values
        return: (B,M,D)
        """
        b = x.shape[0]
        latents = getattr(self, "dynamic_query", None)
        if latents in None:
            latents =  repeat(self.latents, "m d -> b m d", b=b)

        for attn, ff in self.layers:
            if self.gradient_checkpointing and self.training:
                latents = torch.utils.checkpoint.checkpoint(
                    lambda l: attn(x,l)+l, latents, use_reentraint = False
                )
                latents = torch.utils.checkpoint.checkpoint(
                    lambda l: ff(l) + l, latents, use_reentraint = False
                )
            else:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents

        return self.norm(latents)
    
#Anchor selection
def select_anchors_from_cls_attention(
        hidden_states: torch.Tensor,
        attentions_last: torch.Tensor,
        tn: int
) -> torch.Tensor:
    """
    Progressive per-head top-k with deduplication (always include CLS). This CLS is from the output tokens of the vit.
    The top-k selection, choose first row of each attention map in last layer, and from 1st index, they pick total_requireemnts/heads.
    Returns IA (queries): (B, TN+1, D)    
    """

    B,S,D = hidden_states.shape
    H = attentions_last.shape[1]
    N= S= 1
    tn = min(tn, N)

    cls2patch = attentions_last[:,:,0,1:]

    per_head = max(t//H,1)     #target picks per head
    ia_list = []

    for b in range(B):
        chosen = set([0])
        for h in range(H):
            scores = cls2patch[b,h]
            sorted_idx = torch.argsort(Scores) + 1
            need = per_head
            ptr = -1
            while need > 0 and -ptr <= sorted_idx.numel():
                idx = int(sorted_idx[ptr].item())
                if idx not in chosen:
                    chosen.add(idx); need -= 1
                ptr -= 1
            if len(chosen) -1 >= tn:
                break

        # if still short (global fill)
        if len(chosen) -1 < tn:
            global_sorted = (torch.argsort(cls2patch[b].mean(0))+1).tolist()[::-1]
            for idx in global_sorted:
                if idx not in chosen:
                    chosen.add(int(idx))
                    if len(chosen) -1 == tn:
                        break

        take = sorted(list(chosen))
        ia_list.append(hidden_states[b,take, :])

    IA = torch.stack(ia_list, dim = 0)
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
        g =  torch.sigmoid(self,gate(x).squeeze(-1))
        attn = torch.softmax(scores*g, dim=1)
        fused = torch.einsum('bm,bmd->bd', attn, x).unsqueeze(1)
        return fused
    
class AnchorFormerAggregator(nn.Module):
    """
    wrapper: set resampler.dynamic_query = IA, then resample with keys/values R_v
    """
    def __init__(self, d_vision: int, depth:int = 6, dim_head: int=64, heads:int =8):
        super().__init__()
        self.resampler = PereceiverResampler(
            dim = d_vision,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=1,
            ff_mult=4,
            gradient_checkpointing=False
        )
    
    def forward(Self, Rv: torch.Tensor, IA:torch.Tensor)->torch.Tensor:
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
        self.clip = CLIPVisionModel.from_pretrained(af_cfg.vision_model)  # Load CLIP ViT
        if af_cfg.freeze_clip:
            self.clip.requires_grad_(False)              # Freeze weights if requested

        self.processor = None
        if af_cfg.use_processor:
            self.processor = CLIPImageProcessor.from_pretrained(af_cfg.vision_model)  # (optional) preproc

        d_vision = self.clip.config.hidden_size          # CLIP hidden width D (e.g., 768/1024)

        # ---- AnchorFormer aggregator (Perceiver stack) ----
        self.af = AnchorFormerAggregator(
            d_vision=d_vision,
            depth=af_cfg.resampler_depth,
            dim_head=af_cfg.resampler_dim_head,
            heads=af_cfg.resampler_heads,
        )

        # ---- GPA pooling ----
        self.token_gpa = GPAPool(d_in=d_vision, d_hidden=af_cfg.gpa_hidden_size)  # Pool anchors within frame
        self.frame_gpa = GPAPool(d_in=d_vision, d_hidden=af_cfg.gpa_hidden_size)  # Pool across frames

        # ---- Projection to LM width (e.g., T5 d_model) ----
        self.proj = nn.Linear(d_vision, lm_hidden_size)

    def _preprocess_if_needed(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Optionally apply CLIP processor (slow path).
        Prefer doing preprocessing in your dataset/dataloader for speed.
        """
        if not self.cfg.use_processor:
            return imgs                                  # No-op if you preprocess upstream
        # NOTE: CLIPImageProcessor usually expects PIL/np arrays; wiring here is non-trivial.
        # Keep False and handle transforms in your dataset.__getitem__ instead.
        return imgs

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
        )                                                # Shape: (B*T, TN+1, D)

        # ---- Cross-attention aggregation (Perceiver) ----
        Hv = self.af(Rv=Rv, IA=IA)                      # Refined anchors: (B*T, TN+1, D)

        # ---- GPA pooling strategy ----
        # 1) Pool within each frame's anchors -> (B*T, 1, D)
        Hv_pooled = self.token_gpa(Hv)                  # (B*T, 1, D)

        # Reshape back to (B, T, D) by removing the single-token dim
        Hv_pooled = Hv_pooled.view(B, T, 1, -1).squeeze(2)  # (B, T, D)

        # 2) Pool across frames -> (B, 1, D)
        fused = self.frame_gpa(Hv_pooled)               # (B, 1, D)

        # ---- Final projection to LM space ----
        fused_proj = self.proj(fused)                   # (B, 1, lm_hidden_size)
        return fused_proj


