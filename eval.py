import os
import csv
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# ---- GeoFormer stack (your project) ----
from transformers import T5Tokenizer
from dataset import MultiViewDataset
from model import GeoFormer, AnchorFormerConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- utils ----------------
def build_img_tf():
    # Match training’s CLIP-style preprocessing
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

def decode_ignore_pad(tokenizer: T5Tokenizer, ids: torch.Tensor) -> List[str]:
    ids = ids.clone()
    ids[ids == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


# ---- Dependency-free text metrics ----
import re
import string
import math

_ARTICLES = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans("", "", string.punctuation)

def _normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNC_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    # remove leading articles
    toks = s.split()
    toks = [t for t in toks if t not in _ARTICLES]
    return " ".join(toks)

def exact_match(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))

def f1_score(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0

def bleu_4(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p or not r:
        return 0.0
    precisions: List[float] = []
    for n in range(1, 5):
        pn = _ngrams(p, n)
        rn = _ngrams(r, n)
        if not pn:
            precisions.append(0.0)
            continue
        ref_counts: Dict[tuple, int] = {}
        for g in rn:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        match = 0
        used: Dict[tuple, int] = {}
        for g in pn:
            if ref_counts.get(g, 0) - used.get(g, 0) > 0:
                match += 1
                used[g] = used.get(g, 0) + 1
        precisions.append(_safe_div(match, len(pn)))
    if any(p_i == 0.0 for p_i in precisions):
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p_i) for p_i in precisions) / 4.0)
    bp = math.exp(1 - len(r)/len(p)) if len(p) < len(r) else 1.0
    return bp * geo

def rouge_l_f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    # LCS
    dp = [[0]*(len(r)+1) for _ in range(len(p)+1)]
    for i in range(1, len(p)+1):
        for j in range(1, len(r)+1):
            if p[i-1] == r[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    prec = _safe_div(lcs, len(p))
    rec  = _safe_div(lcs, len(r))
    return _safe_div(2*prec*rec, (prec+rec)) if (prec+rec) > 0 else 0.0


# ---------------- prediction ----------------
def collect_predictions(
    model,
    dloader: DataLoader,
    tokenizer: T5Tokenizer,
    max_batches: Optional[int] = None,
    max_new_tokens: int = 32,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns:
      preds: list of dicts with 'caption' (and optional image_id left as None)
      golds: list of ground-truth strings aligned by order
    """
    model.eval()
    preds: List[Dict[str, Any]] = []
    golds: List[str] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(dloader, total=len(dloader), desc="Predict")):
            input_ids, attention_mask, imgs, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            imgs = imgs.to(device)

            # Use GeoFormer.generate; IMPORTANT: pass images=...
            gen_ids = model.generate(
                input_ids,
                attention_mask,
                images=imgs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            captions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            preds.extend([{"image_id": None, "caption": cap.strip()} for cap in captions])

            # decode gold labels
            gold_texts = decode_ignore_pad(tokenizer, labels)
            golds.extend([g.strip() for g in gold_texts])

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    return preds, golds


# --------------- CLI ----------------
def get_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument('--dataset-dir', default=os.path.join('data', 'multi_frame'), type=str)
    p.add_argument('--use-subset', default='none', choices=['none', 'subdata_1', 'subdata_10'],
                   help='If set, read JSONs from data/multi_frame/subsets/<name>/')

    # Model / tokenizer (match train.py)
    p.add_argument('--lm', default='t5-base', choices=['t5-base', 't5-large', 'T5-Base', 'T5-Large'])
    p.add_argument('--gpa-hidden-size', default=512, type=int)
    p.add_argument('--freeze-lm', action='store_true')

    # Checkpoint
    p.add_argument('--checkpoint-dir', required=True,
                   help='Folder inside multi_view_results/<TIMESTR> that has latest_model.pth')
    p.add_argument('--map-location', default='auto', choices=['auto', 'cpu', 'cuda'])

    # Inference
    p.add_argument('--batch-size', default=2, type=int)
    p.add_argument('--num-workers', default=0, type=int)
    p.add_argument('--max-new-tokens', default=32, type=int)

    # Debug
    p.add_argument('--max-batches', type=int, default=None, help='Limit batches for quick smoke test')

    return p.parse_args()


# ---------------- main ----------------
if __name__ == "__main__":
    args = get_args()

    # Resolve data root (supports subsets created by train.py)
    if args.use_subset != 'none':
        data_root = os.path.join(args.dataset_dir, 'subsets', args.use_subset)
    else:
        data_root = args.dataset_dir

    test_json = os.path.join(data_root, 'multi_frame_test.json')
    if not os.path.isfile(test_json):
        raise FileNotFoundError(f"Test JSON not found: {test_json}")

    # Tokenizer (normalize LM name)
    lm_name = args.lm.lower()
    tokenizer = T5Tokenizer.from_pretrained(lm_name)
    # In case your training added this token:
    try:
        tokenizer.add_tokens('<')
    except Exception:
        pass

    # Dataset + loader (ordered; DO NOT shuffle)
    img_tf = build_img_tf()
    test_dset = MultiViewDataset(
        input_file=test_json,
        tokenizer=tokenizer,
        transform=img_tf,
    )
    test_loader = DataLoader(
        test_dset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,                # <- change this to False
        collate_fn=test_dset.collate_fn
    )


    # Model
    af_cfg = AnchorFormerConfig(gpa_hidden_size=args.gpa_hidden_size)
    model = GeoFormer(lm_name=lm_name, af_cfg=af_cfg, freeze_lm=args.freeze_lm).to(device)

    # Load checkpoint
    ck_dir = os.path.join('multi_view_results', args.checkpoint_dir)
    ck_path = os.path.join(ck_dir, 'latest_model.pth')
    if not os.path.isfile(ck_path):
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")

    map_loc = None
    if args.map_location == 'cpu':
        map_loc = torch.device('cpu')
    elif args.map_location == 'cuda':
        map_loc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sd = torch.load(ck_path, map_location=map_loc)
    model.load_state_dict(sd)
    model.eval()

    # Predict (+ collect golds)
    preds, golds = collect_predictions(
        model, test_loader, tokenizer,
        max_batches=args.max_batches,
        max_new_tokens=args.max_new_tokens,
    )

    # Save predictions next to checkpoint
    out_dir = ck_dir
    os.makedirs(out_dir, exist_ok=True)

    # 1) predictions.json (COCO-like but with image_id possibly None)
    results_file = os.path.join(out_dir, 'predictions.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"[done] Wrote predictions: {results_file} (n={len(preds)})")

    # -------- Per-example metrics + aggregates --------
    n = min(len(preds), len(golds))

    per_rows: List[Dict[str, Any]] = []
    em_sum = f1_sum = bleu4_sum = rougeL_sum = 0.0

    for i in range(n):
        pred = preds[i]["caption"]
        gold = golds[i]
        em_i = exact_match(pred, gold)
        f1_i = f1_score(pred, gold)
        bleu4_i = bleu_4(pred, gold)
        rougeL_i = rouge_l_f1(pred, gold)

        em_sum += em_i
        f1_sum += f1_i
        bleu4_sum += bleu4_i
        rougeL_sum += rougeL_i

        per_rows.append({
            "idx": i,
            "image_id": preds[i].get("image_id"),
            "prediction": pred,
            "gold": gold,
            "EM": em_i,
            "F1": round(f1_i, 6),
            "BLEU4": round(bleu4_i, 6),
            "ROUGE_L_F1": round(rougeL_i, 6),
        })

    em = (em_sum / n) if n > 0 else 0.0
    f1 = (f1_sum / n) if n > 0 else 0.0
    bleu4 = (bleu4_sum / n) if n > 0 else 0.0
    rougeL = (rougeL_sum / n) if n > 0 else 0.0

    metrics: Dict[str, Any] = {
        "count": n,
        "EM": em,
        "F1": f1,
        "BLEU4": bleu4,
        "ROUGE_L_F1": rougeL,
    }

    print(f"[local] EM={em:.4f}  F1={f1:.4f}  BLEU4={bleu4:.4f}  ROUGE_L_F1={rougeL:.4f}")

    # 2) predictions_with_gt.jsonl (now includes per-example metrics)
    with_gt_file = os.path.join(out_dir, 'predictions_with_gt.jsonl')
    with open(with_gt_file, 'w', encoding='utf-8') as f:
        for row in per_rows:
            rec = {
                "idx": row["idx"],
                "image_id": row.get("image_id"),
                "prediction": row["prediction"],
                "gold": row["gold"],
                "EM": row["EM"],
                "F1": row["F1"],
                "BLEU4": row["BLEU4"],
                "ROUGE_L_F1": row["ROUGE_L_F1"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] Wrote {with_gt_file}")

    # 3) per_example_metrics.csv (matrix for inspection)
    if per_rows:
        per_csv = os.path.join(out_dir, "per_example_metrics.csv")
        with open(per_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["idx", "image_id", "prediction", "gold", "EM", "F1", "BLEU4", "ROUGE_L_F1"]
            )
            writer.writeheader()
            writer.writerows(per_rows)
        print(f"[done] Wrote per-example table: {per_csv}")

    # 4) metrics.json (always)
    metrics_file = os.path.join(out_dir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] Saved metrics to {metrics_file}")

    # 5) metrics.csv (aggregate, for convenience)
    metrics_csv = os.path.join(out_dir, 'metrics.csv')
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"[done] Saved metrics to {metrics_csv}")
