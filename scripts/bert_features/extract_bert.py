"""
BERT feature extraction for tsmc_clean_filtered.csv (21,068 rows)
Model: bert-base-chinese ([CLS] last hidden state → 768-dim embedding)
Output: data/processed/tsmc_bert_features.npy  (shape: 21068 x 768)

Runtime estimate: ~20-40 min on CPU, ~5 min with GPU.
Checkpoint every 500 batches; resume automatically if interrupted.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizerFast

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "processed" / "tsmc_bert_features.npy"
CKPT_PATH = DATA_DIR / "processed" / "tsmc_bert_features_ckpt.npy"

MODEL_NAME  = "bert-base-chinese"
BATCH_SIZE  = 32
MAX_LEN     = 64    # 64 Chinese characters ≈ headline + first sentence
CKPT_EVERY  = 500   # save checkpoint every N batches


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "processed" / "tsmc_features.csv", encoding="utf-8-sig")
    df["title"]   = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    texts = (df["title"] + " " + df["content"]).str.strip().tolist()
    n = len(texts)
    print(f"  Total articles: {n}")

    # Check checkpoint
    start_idx = 0
    all_embeds = np.zeros((n, 768), dtype=np.float32)
    if CKPT_PATH.exists():
        ckpt = np.load(CKPT_PATH)
        filled = int((ckpt.any(axis=1)).sum())
        if filled > 0:
            all_embeds[:filled] = ckpt[:filled]
            start_idx = filled
            print(f"  Resuming from checkpoint: {filled}/{n} done")

    if start_idx >= n:
        print("Already complete, saving final output.")
        np.save(OUT_PATH, all_embeds)
        return

    print(f"Loading {MODEL_NAME} tokenizer + model...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = BertModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    batches = range(start_idx, n, BATCH_SIZE)
    total_batches = len(range(0, n, BATCH_SIZE))
    done_batches  = start_idx // BATCH_SIZE
    t0 = time.time()

    print(f"Extracting embeddings (batches {done_batches+1}/{total_batches})...")
    with torch.no_grad():
        for i, batch_start in enumerate(batches):
            batch = texts[batch_start: batch_start + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True, truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
            all_embeds[batch_start: batch_start + len(batch)] = cls

            batch_num = done_batches + i + 1
            if (i + 1) % 50 == 0 or (i + 1) == len(batches):
                elapsed = time.time() - t0
                done    = batch_start + len(batch)
                pct     = done / n * 100
                eta     = elapsed / max(i+1, 1) * (len(batches) - i - 1)
                print(f"  [{done:>6}/{n}] {pct:5.1f}% | "
                      f"elapsed={elapsed:.0f}s | ETA~{eta:.0f}s")

            if (i + 1) % CKPT_EVERY == 0:
                np.save(CKPT_PATH, all_embeds)
                print(f"  Checkpoint saved ({batch_start + len(batch)}/{n})")

    np.save(OUT_PATH, all_embeds)
    print(f"\nBERT features saved → {OUT_PATH}")
    print(f"Shape: {all_embeds.shape}  |  Total time: {time.time()-t0:.0f}s")
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()


if __name__ == "__main__":
    main()
