# import os
# import json
# import time
# import argparse
# from copy import deepcopy

# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.transforms import InterpolationMode

# from tqdm import tqdm
# import matplotlib
# matplotlib.use("Agg") 
# import matplotlib.pyplot as plt
# import pandas as pd

# from transformers import T5Tokenizer

# from dataset import MultiViewDataset
# from model import print_trainable_parameters, GeoFormer, AnchorFormerConfig

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.set_num_threads(min(32, os.cpu_count() or 8))   # adjust if you know your core count
# torch.set_num_interop_threads(min(8, os.cpu_count() or 8))


# # ----------------- utils -----------------
# def save_model(obj, model_name, timestr):
#     out_dir = os.path.join('multi_view_results', timestr)
#     os.makedirs(out_dir, exist_ok=True)
#     torch.save(obj, os.path.join(out_dir, f'{model_name}.pth'))

# def validate(dloader, model):
#     model.eval()
#     total = 0.0
#     with torch.no_grad():
#         for batch in tqdm(dloader, total=len(dloader), desc="Valid"):
#             input_ids, attention_mask, imgs, labels = batch
#             outputs = model(input_ids, attention_mask, imgs, labels=labels)
#             total += outputs.loss.item()
#     return total / max(1, len(dloader))

# def save_stats(losses, val_losses, train_min, val_min, epochs, lr, config, timestr):
#     out_dir = os.path.join('multi_view_results', timestr)
#     os.makedirs(out_dir, exist_ok=True)
#     stats_dict = {
#         'losses': losses,
#         'val losses': val_losses,
#         'min train loss': train_min,
#         'min val loss': val_min,
#         'epochs': epochs,
#         'learning rate': lr,
#         'LM': config.lm,
#         'Image Embedding': 'AnchorFormer+GPA'
#     }
#     with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
#         json.dump(stats_dict, f, indent=2)

# def plot_loss(training_loss, val_loss, timestr):
#     out_dir = os.path.join('multi_view_results', timestr)
#     os.makedirs(out_dir, exist_ok=True)
#     plt.figure()
#     plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
#     plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(os.path.join(out_dir, 'loss.png'))
#     plt.close()

# def decode_ignore_pad(tokenizer, ids):
#     ids = ids.clone()
#     ids[ids == -100] = tokenizer.pad_token_id
#     return tokenizer.batch_decode(ids, skip_special_tokens=True)

# # --------------- training ----------------
# def custom_train(model, train_dataloader, val_dataloader, tokenizer, timestr, config):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#     losses, val_losses = [], []
#     best_model_sd, best_val = None, float("inf")
#     epochs_run = 0

#     for epoch in range(config.epochs):
#         print(f"\n------------- EPOCH {epoch+1}/{config.epochs} -------------")
#         model.train()
#         epoch_loss = 0.0

#         for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader), desc="Train"), start=1):
#             input_ids, attention_mask, imgs, labels = batch

#             # one-time shape sanity on very first batch
#             if epoch == 0 and step == 1:
#                 print(f"[diag] input_ids {tuple(input_ids.shape)} | attention_mask {tuple(attention_mask.shape)}")
#                 print(f"[diag] imgs {tuple(imgs.shape)} | labels {tuple(labels.shape)}")

#             outputs = model(input_ids, attention_mask, imgs, labels=labels)
#             loss = outputs.loss

#             # quick NaN/Inf guard
#             if not torch.isfinite(loss):
#                 raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}: {loss.item()}")

#             epoch_loss += loss.item()

#             if step % config.checkpoint_frequency == 0 or (epoch == 0 and step == 1):
#                 with torch.no_grad():
#                     # rough preview via argmax (fast). For proper eval, use generate().
#                     pred_ids = torch.argmax(outputs.logits, dim=-1)
#                     q_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#                     tgt_texts = decode_ignore_pad(tokenizer, labels)
#                     pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#                     print(f"\n[step {step}] loss={loss.item():.4f}")
#                     print("[Q]   ", q_texts[0] if q_texts else "")
#                     print("[pred]", pred_texts[0] if pred_texts else "")
#                     print("[gold]", tgt_texts[0] if tgt_texts else "")

#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()

#             # one-time grad norm sanity
#             if epoch == 0 and step == 1:
#                 gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
#                 print(f"[diag] grad_norm={float(gnorm):.2f}")

#             optimizer.step()

#         # epoch end
#         tr = epoch_loss / max(1, len(train_dataloader))
#         losses.append(tr)
#         val = validate(val_dataloader, model)
#         val_losses.append(val)
#         scheduler.step()

#         # track best
#         if val < best_val:
#             best_val = val
#             best_model_sd = deepcopy(model.state_dict())
#             save_model(best_model_sd, 'latest_model', timestr)

#         epochs_run += 1
#         print(f"epoch train={tr:.4f} | val={val:.4f}")

#     plot_loss(losses, val_losses, timestr)
#     save_stats(losses, val_losses, min(losses), min(val_losses), epochs_run, scheduler.get_last_lr()[0], config, timestr)
#     return min(losses), min(val_losses)

# def params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--learning-rate", default=1e-4, type=float)
#     parser.add_argument("--batch-size", default=2, type=int)
#     parser.add_argument("--weight-decay", default=0.05, type=float)
#     parser.add_argument("--epochs", default=5, type=int)
#     parser.add_argument('--gpa-hidden-size', default=512, type=int)
#     parser.add_argument('--freeze-lm', action='store_true')
#     parser.add_argument('--lm', default='t5-base', choices=['t5-base', 't5-large', 'T5-Base', 'T5-Large'], type=str)
#     parser.add_argument('--checkpoint-frequency', default=200, type=int)
#     parser.add_argument('--num-workers', default=0, type=int)
#     parser.add_argument('--load-checkpoint', action='store_true')
#     parser.add_argument('--checkpoint-file', default='', type=str)
#     return parser.parse_args()

# # ------------------- main -------------------
# if __name__ == '__main__':
#     timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
#     print(f"Run directory: multi_view_results/{timestr}")
#     config = params()

#     # normalize LM name (accepts T5-Base / t5-base)
#     lm_name = config.lm.lower()

#     # tokenizer
#     tokenizer = T5Tokenizer.from_pretrained(lm_name)

#     # CLIP ViT preprocessing for read_image (uint8 -> float32 [0,1] -> normalize)
#     img_tf = transforms.Compose([
#         transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
#         transforms.ConvertImageDtype(torch.float32),
#         transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
#                              std=(0.26862954, 0.26130258, 0.27577711)),
#     ])

#     # datasets
#     train_dset = MultiViewDataset(
#         input_file=os.path.join('data', 'multi_frame', 'multi_frame_train.json'),
#         tokenizer=tokenizer,
#         transform=img_tf,
#     )
#     val_dset = MultiViewDataset(
#         input_file=os.path.join('data', 'multi_frame', 'multi_frame_val.json'),
#         tokenizer=tokenizer,
#         transform=img_tf,
#     )
#     test_dset = MultiViewDataset(
#         input_file=os.path.join('data', 'multi_frame', 'multi_frame_test.json'),
#         tokenizer=tokenizer,
#         transform=img_tf,
#     )

#     train_dataloader = DataLoader(train_dset, shuffle=True, batch_size=config.batch_size,
#                                   num_workers=config.num_workers, pin_memory=(device.type == 'cuda'),
#                                   collate_fn=train_dset.collate_fn)
#     val_dataloader = DataLoader(val_dset, shuffle=False, batch_size=config.batch_size,
#                                 num_workers=config.num_workers, pin_memory=(device.type == 'cuda'),
#                                 collate_fn=val_dset.collate_fn)
#     test_dataloader = DataLoader(test_dset, shuffle=False, batch_size=config.batch_size,
#                                  num_workers=config.num_workers, pin_memory=(device.type == 'cuda'),
#                                  collate_fn=test_dset.collate_fn)

#     # model
#     af_cfg = AnchorFormerConfig(gpa_hidden_size=config.gpa_hidden_size)
#     model = GeoFormer(lm_name=lm_name, af_cfg=af_cfg, freeze_lm=config.freeze_lm).to(device)
#     print('Trainable Parameters for full model')
#     print_trainable_parameters(model)

#     tokenizer.add_tokens('<')  

#     # ----- checkpoint load (optional) -----
#     if config.load_checkpoint and config.checkpoint_file:
#         ck_dir = os.path.join('multi_view_results', config.checkpoint_file)
#         print(f'Loading checkpoint from {ck_dir}')
#         sd = torch.load(os.path.join(ck_dir, 'latest_model.pth'), map_location=device)
#         model.load_state_dict(sd)

#     # ----- train -----
#     min_train_loss, min_val_loss = custom_train(
#         model, train_dataloader, val_dataloader, tokenizer, timestr, config
#     )

#     # ----- test -----
#     # reload best ckpt if it exists
#     ck_dir = os.path.join('multi_view_results', timestr)
#     best_model = GeoFormer(lm_name=lm_name, af_cfg=af_cfg, freeze_lm=config.freeze_lm).to(device)
#     latest_ckpt = os.path.join(ck_dir, 'latest_model.pth')
#     if os.path.isfile(latest_ckpt):
#         best_model.load_state_dict(torch.load(latest_ckpt, map_location=device))

#     test_loss = validate(test_dataloader, best_model)

#     # record trial
#     trial_dict = {
#         'Model name': [timestr],
#         'Learning rate': [config.learning_rate],
#         'Weight decay': [config.weight_decay],
#         'Batch size': [config.batch_size],
#         'Epochs': [config.epochs],
#         'GPA Hidden Size': [config.gpa_hidden_size],
#         'Freeze T5': [config.freeze_lm],
#         'Min Training Loss': [min_train_loss],
#         'Min Validation Loss': [min_val_loss],
#         'Min Testing Loss': [test_loss],
#     }
#     df = pd.DataFrame(trial_dict)
#     out_dir = os.path.join('multi_view_results', timestr)
#     os.makedirs(out_dir, exist_ok=True)
#     df.to_csv(os.path.join(out_dir, 'multi_view_results.csv'), index=False, header=True)

#     print("\nDone. Best ckpt @", latest_ckpt if os.path.isfile(latest_ckpt) else "(not saved)")




import os
import json
import time
import argparse
from copy import deepcopy
import random
from typing import Tuple, List, Literal, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from transformers import T5Tokenizer

from dataset import MultiViewDataset
from model import print_trainable_parameters, GeoFormer, AnchorFormerConfig

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
_HAS_PEFT = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_num_threads(min(32, os.cpu_count() or 8))   # adjust if you know your core count
torch.set_num_interop_threads(min(8, os.cpu_count() or 8))

JsonFmt = Literal["json", "jsonl"]

def _load_json_flexible(path: str) -> Tuple[List[dict], JsonFmt]:
    """
    Load either:
      - a single JSON array file, or
      - JSONL (one JSON object per line).
    Returns (records, format).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Try array JSON
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data, "json"
    except json.JSONDecodeError:
        pass

    # Fallback to JSONL
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    recs = [json.loads(ln) for ln in lines]
    return recs, "jsonl"


def _save_like(src_fmt: JsonFmt, out_path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if src_fmt == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sample_records(recs: List[dict], percent: int, seed: int) -> List[dict]:
    if percent <= 0:
        return []
    rnd = random.Random(seed)
    k = max(1, int(len(recs) * percent / 100.0))
    if k >= len(recs):
        return recs
    idxs = rnd.sample(range(len(recs)), k)
    return [recs[i] for i in idxs]


def _subset_name_to_percent(name: str) -> int:
    # "subdata_1" -> 1, "subdata_10" -> 10
    if name == "subdata_1":
        return 1
    if name == "subdata_10":
        return 10
    raise ValueError(f"Unknown subset name: {name}")


def _create_single_subset(
    dataset_dir: str,
    subset_name: str,
    seed: int,
    splits: Dict[str, str] = None
) -> Dict[str, int]:
    """
    Create one subset under dataset_dir/subsets/{subset_name}/
    Returns counts per split.
    """
    if splits is None:
        splits = {
            "train": "multi_frame_train.json",
            "val":   "multi_frame_val.json",
            "test":  "multi_frame_test.json",
        }
    percent = _subset_name_to_percent(subset_name)
    subset_dir = os.path.join(dataset_dir, "subsets", subset_name)
    os.makedirs(subset_dir, exist_ok=True)

    counts = {}
    for split, fname in splits.items():
        src_file = os.path.join(dataset_dir, fname)
        if not os.path.isfile(src_file):
            raise FileNotFoundError(f"Source split not found: {src_file}")

        recs, fmt = _load_json_flexible(src_file)
        sub_recs = _sample_records(recs, percent, seed)
        dst_file = os.path.join(subset_dir, fname)
        _save_like(fmt, dst_file, sub_recs)
        counts[split] = len(sub_recs)

    # also drop a small meta file
    meta = {
        "subset_name": subset_name,
        "percent": percent,
        "seed": seed,
        "counts": counts,
    }
    with open(os.path.join(subset_dir, "subset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return counts


def ensure_subset(dataset_dir: str, subset_name: str, seed: int) -> Dict[str, int]:
    """
    If subset doesn't exist (missing any split file), create it from full set.
    Returns counts.
    """
    subset_dir = os.path.join(dataset_dir, "subsets", subset_name)
    required = [
        os.path.join(subset_dir, "multi_frame_train.json"),
        os.path.join(subset_dir, "multi_frame_val.json"),
        os.path.join(subset_dir, "multi_frame_test.json"),
    ]
    if all(os.path.isfile(p) for p in required):
        # Try load meta if present
        meta_path = os.path.join(subset_dir, "subset_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("counts", {})
        else:
            # Count by reading files
            counts = {}
            for lbl, path in zip(["train", "val", "test"], required):
                recs, _ = _load_json_flexible(path)
                counts[lbl] = len(recs)
            return counts
    # Need to (re)create
    return _create_single_subset(dataset_dir, subset_name, seed)


# ----------------- utils -----------------
def save_model(obj, model_name, timestr):
    out_dir = os.path.join('multi_view_results', timestr)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(obj, os.path.join(out_dir, f'{model_name}.pth'))

def validate(dloader, model):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(dloader, total=len(dloader), desc="Valid"):
            input_ids, attention_mask, imgs, labels = batch
            outputs = model(input_ids, attention_mask, imgs, labels=labels)
            total += outputs.loss.item()
    return total / max(1, len(dloader))

def save_stats(losses, val_losses, train_min, val_min, epochs, lr, config, timestr):
    out_dir = os.path.join('multi_view_results', timestr)
    os.makedirs(out_dir, exist_ok=True)
    stats_dict = {
        'losses': losses,
        'val losses': val_losses,
        'min train loss': train_min,
        'min val loss': val_min,
        'epochs': epochs,
        'learning rate': lr,
        'LM': config.lm,
        'Image Embedding': 'AnchorFormer+GPA',
        'subset_used': config.use_subset,
    }
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)

def plot_loss(training_loss, val_loss, timestr):
    out_dir = os.path.join('multi_view_results', timestr)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

def decode_ignore_pad(tokenizer, ids):
    ids = ids.clone()
    ids[ids == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(ids, skip_special_tokens=True)

def apply_lora_to_t5(geo_model, r: int, alpha: int, dropout: float, targets):
    if not _HAS_PEFT:
        raise RuntimeError("peft is not installed. Run: pip install peft")
    # Typical T5 targets are q,k,v,o (attention projections). You can add 'wi','wo' to hit FFN too.
    lconf = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )
    geo_model.lm = get_peft_model(geo_model.lm, lconf)
    try:
        geo_model.lm.print_trainable_parameters()
    except Exception:
        pass
    return geo_model


# --------------- training ----------------
def custom_train(model, train_dataloader, val_dataloader, tokenizer, timestr, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    losses, val_losses = [], []
    best_model_sd, best_val = None, float("inf")
    epochs_run = 0

    for epoch in range(config.epochs):
        print(f"\n------------- EPOCH {epoch+1}/{config.epochs} -------------")
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader), desc="Train"), start=1):
            input_ids, attention_mask, imgs, labels = batch

            # one-time shape sanity on very first batch
            if epoch == 0 and step == 1:
                print(f"[diag] input_ids {tuple(input_ids.shape)} | attention_mask {tuple(attention_mask.shape)}")
                print(f"[diag] imgs {tuple(imgs.shape)} | labels {tuple(labels.shape)}")

            outputs = model(input_ids, attention_mask, imgs, labels=labels)
            loss = outputs.loss

            # quick NaN/Inf guard
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}: {loss.item()}")

            epoch_loss += loss.item()

            if step % config.checkpoint_frequency == 0 or (epoch == 0 and step == 1):
                with torch.no_grad():
                    # rough preview via argmax (fast). For proper eval, use generate().
                    pred_ids = torch.argmax(outputs.logits, dim=-1)
                    q_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    tgt_texts = decode_ignore_pad(tokenizer, labels)
                    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                    print(f"\n[step {step}] loss={loss.item():.4f}")
                    print("[Q]   ", q_texts[0] if q_texts else "")
                    print("[pred]", pred_texts[0] if pred_texts else "")
                    print("[gold]", tgt_texts[0] if tgt_texts else "")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # one-time grad norm sanity
            if epoch == 0 and step == 1:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
                print(f"[diag] grad_norm={float(gnorm):.2f}")

            optimizer.step()

        # epoch end
        tr = epoch_loss / max(1, len(train_dataloader))
        losses.append(tr)
        val = validate(val_dataloader, model)
        val_losses.append(val)
        scheduler.step()

        # track best
        if val < best_val:
            best_val = val
            # If LoRA is active, save a merged checkpoint to keep eval simple.
            if _HAS_PEFT and hasattr(model, "lm") and isinstance(getattr(model, "lm"), PeftModel):
                # Save adapter (optional)
                if getattr(config, "lora_save_adapter", False):
                    adapter_dir = os.path.join('multi_view_results', timestr, 'lora_adapter')
                    os.makedirs(adapter_dir, exist_ok=True)
                    model.lm.save_pretrained(adapter_dir)
                # Save MERGED weights to latest_model.pth (drop-in for eval.py)
                merged = deepcopy(model).to('cpu')
                merged.lm = merged.lm.merge_and_unload()
                best_model_sd = merged.state_dict()
            else:
                best_model_sd = deepcopy(model.state_dict())
            save_model(best_model_sd, 'latest_model', timestr)


        epochs_run += 1
        print(f"epoch train={tr:.4f} | val={val:.4f}")

    plot_loss(losses, val_losses, timestr)
    save_stats(losses, val_losses, min(losses), min(val_losses), epochs_run, scheduler.get_last_lr()[0], config, timestr)
    return min(losses), min(val_losses)

def params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument('--gpa-hidden-size', default=512, type=int)
    parser.add_argument('--freeze-lm', action='store_true')
    parser.add_argument('--lm', default='t5-base', choices=['t5-base', 't5-large', 'T5-Base', 'T5-Large'], type=str)
    parser.add_argument('--checkpoint-frequency', default=200, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--load-checkpoint', action='store_true')
    parser.add_argument('--checkpoint-file', default='', type=str)

    # -------- Subset controls (NEW) --------
    parser.add_argument('--dataset-dir', default=os.path.join('data', 'multi_frame'), type=str,
                        help='Directory containing full JSONs (multi_frame_*).')
    parser.add_argument('--use-subset', default='none', choices=['none', 'subdata_1', 'subdata_10'],
                        help='Train/eval on this subset (auto-creates if missing).')
    parser.add_argument('--create-subsets', action='store_true',
                        help='Create default subsets (1%% and 10%%) from full data.')
    parser.add_argument('--seed', default=13, type=int, help='Seed for deterministic subset sampling.')
    parser.add_argument('--dry-run', action='store_true',
                        help='If set with --create-subsets, create/check subsets then exit.')
    
        # -------- LoRA controls (NEW) --------
    parser.add_argument('--use-lora', action='store_true',
                        help='Enable LoRA on the T5 LM (model.lm)')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora-target', nargs='*', default=['q','k','v','o'],
                        help='Target module names inside T5Attention to inject LoRA into')
    parser.add_argument('--lora-save-adapter', action='store_true',
                        help='Also save LoRA adapter weights to lora_adapter/ alongside latest_model.pth')

    return parser.parse_args()

# ------------------- main -------------------
if __name__ == '__main__':
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Run directory: multi_view_results/{timestr}")
    config = params()

    # ===== Handle subset creation / selection =====
    # If asked to create subsets, do it up front.
    if config.create_subsets:
        print("[subset] Creating default subsets: subdata_1 (1%) and subdata_10 (10%)")
        c1 = ensure_subset(config.dataset_dir, "subdata_1", config.seed)
        c10 = ensure_subset(config.dataset_dir, "subdata_10", config.seed)
        print(f"[subset] subdata_1 counts: {c1}")
        print(f"[subset] subdata_10 counts: {c10}")
        if config.dry_run and config.use_subset == "none":
            print("[subset] Dry-run requested. Exiting after subset creation.")
            raise SystemExit(0)

    # Decide which directory to read JSONs from
    if config.use_subset != 'none':
        subset_dir = os.path.join(config.dataset_dir, "subsets", config.use_subset)
        # Auto-create if missing
        counts = ensure_subset(config.dataset_dir, config.use_subset, config.seed)
        print(f"[subset] Using {config.use_subset} at {subset_dir} with counts: {counts}")
        data_root = subset_dir
    else:
        data_root = config.dataset_dir
        print(f"[data] Using full dataset at: {data_root}")

    # normalize LM name (accepts T5-Base / t5-base)
    lm_name = config.lm.lower()

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(lm_name)

    # CLIP ViT preprocessing for read_image (uint8 -> float32 [0,1] -> normalize)
    img_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # datasets
    # Note: filenames preserved whether full or subset
    train_path = os.path.join(data_root, 'multi_frame_train.json')
    val_path   = os.path.join(data_root, 'multi_frame_val.json')
    test_path  = os.path.join(data_root, 'multi_frame_test.json')

    if not (os.path.isfile(train_path) and os.path.isfile(val_path) and os.path.isfile(test_path)):
        raise FileNotFoundError(
            f"Could not find one or more dataset files at:\n  {train_path}\n  {val_path}\n  {test_path}"
        )

    train_dset = MultiViewDataset(
        input_file=train_path,
        tokenizer=tokenizer,
        transform=img_tf,
    )
    val_dset = MultiViewDataset(
        input_file=val_path,
        tokenizer=tokenizer,
        transform=img_tf,
    )
    test_dset = MultiViewDataset(
        input_file=test_path,
        tokenizer=tokenizer,
        transform=img_tf,
    )

    train_dataloader = DataLoader(train_dset, shuffle=True, batch_size=config.batch_size,
                                  num_workers=config.num_workers, pin_memory=(device.type == 'cuda'),
                                  collate_fn=train_dset.collate_fn)
    val_dataloader = DataLoader(val_dset, shuffle=False, batch_size=config.batch_size,
                                num_workers=config.num_workers, pin_memory=(device.type == 'cuda'),
                                collate_fn=val_dset.collate_fn)
    test_dataloader = DataLoader(test_dset, shuffle=False, batch_size=config.batch_size,
                                 num_workers=config.num_workers, pin_memory=(device.type == 'cuda'),
                                 collate_fn=test_dset.collate_fn)

    # model
    af_cfg = AnchorFormerConfig(gpa_hidden_size=config.gpa_hidden_size)
    model = GeoFormer(lm_name=lm_name, af_cfg=af_cfg, freeze_lm=config.freeze_lm).to(device)

    # ----- LoRA (optional) -----
    if config.use_lora:
        if not _HAS_PEFT:
            raise RuntimeError("You enabled --use-lora but peft is missing. pip install peft")
        if config.freeze_lm:
            print("[warn] --freeze-lm is ignored for LoRA; base LM weights are frozen by PEFT automatically.")
        model = apply_lora_to_t5(
            model,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            targets=config.lora_target,
        )

    print('Trainable Parameters for full model')
    print_trainable_parameters(model)

    tokenizer.add_tokens('<')

    # ----- checkpoint load (optional) -----
    if config.load_checkpoint and config.checkpoint_file:
        ck_dir = os.path.join('multi_view_results', config.checkpoint_file)
        print(f'Loading checkpoint from {ck_dir}')
        sd = torch.load(os.path.join(ck_dir, 'latest_model.pth'), map_location=device)
        model.load_state_dict(sd)

    # If user only wanted subsets and asked to dry-run while still picking a subset,
    # continue. Otherwise, proceed to training.
    # ----- train -----
    min_train_loss, min_val_loss = custom_train(
        model, train_dataloader, val_dataloader, tokenizer, timestr, config
    )

    # ----- test -----
    # reload best ckpt if it exists
    ck_dir = os.path.join('multi_view_results', timestr)
    best_model = GeoFormer(lm_name=lm_name, af_cfg=af_cfg, freeze_lm=config.freeze_lm).to(device)
    latest_ckpt = os.path.join(ck_dir, 'latest_model.pth')
    if os.path.isfile(latest_ckpt):
        best_model.load_state_dict(torch.load(latest_ckpt, map_location=device))

    test_loss = validate(test_dataloader, best_model)

    # record trial
    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [config.learning_rate],
        'Weight decay': [config.weight_decay],
        'Batch size': [config.batch_size],
        'Epochs': [config.epochs],
        'GPA Hidden Size': [config.gpa_hidden_size],
        'Freeze T5': [config.freeze_lm],
        'Subset': [config.use_subset],
        'Min Training Loss': [min_train_loss],
        'Min Validation Loss': [min_val_loss],
        'Min Testing Loss': [test_loss],
    }
    df = pd.DataFrame(trial_dict)
    out_dir = os.path.join('multi_view_results', timestr)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'multi_view_results.csv'), index=False, header=True)

    print("\nDone. Best ckpt @", latest_ckpt if os.path.isfile(latest_ckpt) else "(not saved)")
