import os, json, torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import T5Tokenizer

from dataset import MultiViewDataset
from model import GeoFormer, AnchorFormerConfig

# ---------- setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dir = r"multi_view_results\2025-09-12_21-39-57"
ckpt = os.path.join(ckpt_dir, "latest_model.pth")

lm_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(lm_name)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

img_tf = transforms.Compose([
    transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

# ---------- data ----------
INPUT_JSON = os.path.join("data", "multi_frame", "multi_frame_test.json")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw_items = json.load(f)  # each item: [ qa_dict, cam_dict ]

ds = MultiViewDataset(input_file=INPUT_JSON, tokenizer=tokenizer, transform=img_tf)

# ---------- model ----------
af_cfg = AnchorFormerConfig(gpa_hidden_size=512)
model = GeoFormer(lm_name=lm_name, af_cfg=af_cfg, freeze_lm=False).to(device)
state = torch.load(ckpt, map_location=device)
model.load_state_dict(state)
model.eval()

# ---------- helpers ----------
def decode_ignore_pad(tok: T5Tokenizer, ids: torch.Tensor) -> str:
    ids = ids.clone()
    ids[ids == -100] = tok.pad_token_id
    return tok.batch_decode(ids, skip_special_tokens=True)[0].strip()

CAM_ORDER = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

# Repo root (where 'data/' lives) and MD dir (where report is written)
REPO_ROOT = os.path.abspath(os.getcwd())
MD_DIR    = os.path.abspath(ckpt_dir)

def resolve_abs(p: str) -> str:
    """Resolve JSON path to an absolute file under the repo root."""
    if not p:
        return ""
    p = p.replace("\\", "/")
    # Typical case: 'data/...'
    if p.startswith("data/"):
        cand = os.path.join(REPO_ROOT, p)
        return os.path.abspath(cand) if os.path.exists(cand) else ""
    # Absolute path already?
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # Fallback: try relative to repo root
    cand2 = os.path.join(REPO_ROOT, p)
    return os.path.abspath(cand2) if os.path.exists(cand2) else ""

def rel_for_md(abs_path: str) -> str:
    """Make a Markdown-friendly (and HTML-friendly) relative path from MD_DIR to abs_path."""
    rel = os.path.relpath(abs_path, start=MD_DIR).replace("\\", "/")
    return rel  # IMPORTANT: no angle brackets here

def get_cam_pairs(idx: int):
    """
    Return [(CAM_NAME, abs_path or ''), ...] in CAM_ORDER.
    Also print a warning for any missing file.
    """
    rec = raw_items[idx]
    cam_dict = rec[1] if isinstance(rec, (list, tuple)) and len(rec) >= 2 and isinstance(rec[1], dict) else {}
    pairs, missing = [], []
    for cam in CAM_ORDER:
        p_json = (cam_dict.get(cam, "") or "").strip()
        abs_p = resolve_abs(p_json)
        if abs_p:
            pairs.append((cam, abs_p))
        else:
            pairs.append((cam, ""))
            missing.append((cam, p_json))
    if missing:
        print(f"[warn] idx={idx} missing:", ", ".join([f"{c}:{r}" for c, r in missing]))
    return pairs

def views_table_md(cam_pairs):
    """
    Render a 2x(<=3 cols) grid with camera names and images.
    Uses proper quoted src attributes.
    """
    names, rel_paths = [], []
    for cam, abs_p in cam_pairs:
        if abs_p and os.path.exists(abs_p):
            names.append(cam)
            rel_paths.append(rel_for_md(abs_p))

    if not rel_paths:
        return "**Views:** (no image paths found on disk)"

    # Build rows in chunks of 3 columns
    rows = ["<table>"]
    for i in range(0, len(names), 3):
        n_chunk = names[i:i+3]
        p_chunk = rel_paths[i:i+3]
        # Header row
        rows.append("<tr>")
        for n in n_chunk:
            rows.append(f"<th style='text-align:center;padding:4px'>{n}</th>")
        rows.append("</tr>")
        # Image row
        rows.append("<tr>")
        for p in p_chunk:
            rows.append(
                f"<td style='text-align:center;padding:4px'>"
                f"<img src=\"{p}\" alt=\"\" width=\"340\"/></td>"
            )
        rows.append("</tr>")
    rows.append("</table>")
    return "\n".join(rows)

# ---------- outputs ----------
md_path = os.path.join(ckpt_dir, "results_preview.md")

# ---------- run a few examples ----------
n_examples = min(5, len(ds))
rows_md = []

with torch.inference_mode():
    for idx in range(n_examples):
        q_texts, input_ids, attention_mask, imgs, labels, _ = ds.test_collate_fn([ds[idx]])

        out_ids = model.generate(
            input_ids.to(device),
            attention_mask.to(device),
            images=imgs.to(device),
            max_new_tokens=8,
            num_beams=1,
            do_sample=False
        )
        pred = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        gold = decode_ignore_pad(tokenizer, labels)

        cam_pairs = get_cam_pairs(idx)        # [(CAM_..., ABS_PATH or ''), ...]
        table_md  = views_table_md(cam_pairs)  # uses quoted src with rel paths

        md = []
        md.append(f"## Example {idx}\n")
        md.append(f"**Q:** {q_texts[0]}")
        md.append(f"**Pred:** `{pred}`")
        md.append(f"**Gold:** `{gold}`\n")
        md.append(table_md)
        md.append("")
        rows_md.append("\n".join(md))

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# GeoFormer — Preview (Q/Pred/Gold + Named Views)\n\n")
    f.write("\n\n".join(rows_md))

print(f"\n[done] Wrote Markdown report: {md_path}")
print(f"[tip] Open it in VS Code (Ctrl+Shift+V). Images are referenced relative to: {MD_DIR}")
