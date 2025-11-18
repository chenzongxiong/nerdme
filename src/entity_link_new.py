# Standard library
import time
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
import requests
from huggingface_hub import list_datasets
import torch
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
from base_predictor import LABELS
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_entity_from_tsv(files: List[Path], target_label: str):
    mentions = []

    for file in files:
        doc = webanno_tsv_read_file(file)
        for annotation in doc.annotations:
            if annotation.label != target_label:
                continue

            sentences = doc.annotation_sentences(annotation)
            for sentence in sentences:
                mentions.append({
                    "sentence": sentence.text,
                    "entity_text": annotation.text,
                    "entity_type": annotation.label
                })

    df = pd.DataFrame(mentions)
    return df

def fetch_zenodo_records(query, max_records=200):
    """Fetch a few hundred records from Zenodo for canonical linking."""
    url = "https://zenodo.org/api/records"
    page = 1
    all_items = []
    # while len(all_items) < max_records:
    if True:
        r = requests.get(url, params={"q": query, "size": 10, "page": page})
        hits = r.json().get("hits", {}).get("hits", [])
        # if not hits:
        #     break
        for h in hits:
            md = h.get("metadata", {})
            title = md.get("title", "")
            doi = md.get("doi", "")
            resource_type = md.get("resource_type", {}).get("type", "")
            if resource_type in ("dataset", "software"):
                all_items.append({
                    "canonical_id": str(h.get("id", "")),
                    "name": title.strip(),
                    "type": resource_type.capitalize(),
                    "aliases": title.lower(),
                    "gold_name": query,
                    "homepage": h.get("links", {}).get("html", ""),
                    "doi": doi,
                    "source": "Zenodo"
                })
        page += 1

    # df = pd.DataFrame(all_items)
    # return df
    return all_items

def semantic_matching(detected_entities, canonical_entities, *, TOPK, THRESH):
    # 2) Pre-encode canonical entities once (normalized embeddings)
    pred_rows = []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to(device)
    detected_emb = model.encode(detected_entities, convert_to_tensor=True)
    canonical_emb = model.encode(canonical_entities, convert_to_tensor=True)
    cosine_scores = util.cos_sim(detected_emb, canonical_emb)
    for i, ent in enumerate(detected_entities):
        topk = cosine_scores[i].argsort().numpy()[::-1][:TOPK]
        best_match_id = topk[0]
        score = cosine_scores[i][best_match_id].item()
        topk_names = [canonical_entities[x] for x in topk]
        pred_name = canonical_entities[best_match_id]
        if score >= THRESH:
            # print(f"{ent} → {pred_name} ({score:.2f})")
            pred_rows.append({
                "entity_text": ent,
                "pred_name": pred_name,
                "score": score,     # cosine in [-1,1]; typically 0.0–0.9 for good matches
                "candidates": topk_names,
            })
        else:
            pred_rows.append({
                "entity_text": ent,
                "pred_name": 'NIL',
                "score": score,     # cosine in [-1,1]; typically 0.0–0.9 for good matches
                "candidates": topk_names
            })
    pred_df = pd.DataFrame(pred_rows)
    return pred_df

def keyword_matching(detected_entities, canonical_entities, *, TOPK, THRESH):
    pred_rows = []
    for ent in tqdm(detected_entities, desc="Linking (fuzzy)"):
        # top-1
        best = process.extractOne(ent, canonical_entities, scorer=fuzz.token_sort_ratio)
        # top-k (for Hits@k/MRR). Returns list of tuples (match, score, idx)
        topk = process.extract(ent, canonical_entities, scorer=fuzz.token_sort_ratio, limit=TOPK)
        if best is None:
            pred_name, score = 'NIL', 0
            topk_names = []
        else:
            pred_name, score, _ = best
            if score < THRESH * 100:
                pred_name = 'NIL'
            topk_names = [m[0] for m in topk] if topk else []

        pred_rows.append({
            'entity_text': ent,
            'pred_name': pred_name,
            'score': score,
            'candidates': topk_names
        })
    pred_df = pd.DataFrame(pred_rows)
    return pred_df


def main(args):
    # Prepare NERdME entity
    # files = [x for x in Path("./data/train").rglob("*.tsv")] + [x for x in Path("./data/val").rglob("*.tsv")] + [x for x in Path("./data/test_labeled/").rglob("*.tsv")]
    # for target_label in LABELS:
    #     df = extract_entity_from_tsv(files, target_label=target_label)
    #     output_csv = Path(f'./results/entity-linking/nerdeme/{target_label}.csv')
    #     output_csv.parent.mkdir(parents=True, exist_ok=True)
    #     print(f"✅ Extracted {len(df)} {target_label} entities → {output_csv}")
    #     df.to_csv(output_csv)

    target_label = args.target_label
    gold_path = Path('./results/entity-linking/nerdme') / f'{target_label}_gold.csv'

    nerdme_entity_path = Path(f'./results/entity-linking/nerdme/{target_label}.csv')
    nerdme_df = pd.read_csv(nerdme_entity_path)

    # Attach gold if available; otherwise default NIL
    if gold_path.exists():
        gold_df = pd.read_csv(gold_path)  # columns: entity_text, gold_name  (gold_name in canonical_entities or 'NIL')
        nerdme_df = nerdme_df.merge(gold_df[['entity_text','gold_name']], on='entity_text', how='left')
        nerdme_df['gold_name'] = nerdme_df['gold_name'].apply(lambda x: x.lower())
    else:
        nerdme_df['gold_name'] = 'NIL'

    detected_entities = nerdme_df['entity_text'].tolist()
    all_items = []
    queried_set = set()
    ckpt_path = Path('./results/entity-linking/zenodo/.ckpt')
    import ipdb; ipdb.set_trace()



    # if ckpt_path.exists():
    #     saved_idx = int(ckpt_path.read_text())
    #     output_csv = Path(f"./results/entity-linking/zenodo/{target_label}.csv")
    #     df = pd.read_csv(output_csv)
    #     for i, row in df.iterrows():
    #         all_items.append(dict(row))
    # else:
    #     saved_idx = 0

    # for idx, query in enumerate(detected_entities):
    #     if idx < saved_idx:
    #         continue

    #     if query.lower() in queried_set:
    #         continue
    #     queried_set.add(query.lower())

    #     items = fetch_zenodo_records(query.lower(), max_records=10)
    #     all_items.extend(items)
    #     output_csv = Path(f"./results/entity-linking/zenodo/{target_label}.csv")
    #     output_csv.parent.mkdir(parents=True, exist_ok=True)
    #     time.sleep(0.3)
    #     df = pd.DataFrame(all_items)
    #     df.to_csv(output_csv, index=False)
    #     logging.info(f"Saved {len(df)} Zenodo canonical entries to {output_csv}")
    #     ckpt_path.write_text(str(idx))

    # datasets = list_datasets(limit=1000)
    # rows = []

    # for ds in datasets:
    #     rows.append({
    #         "canonical_id": ds.id.lower().replace("/", "_"),
    #         "name": ds.id.split("/")[-1],
    #         "type": "Dataset",
    #         "aliases": ds.id,
    #         "homepage": f"https://huggingface.co/datasets/{ds.id}",
    #         "source": "HuggingFace Hub"
    #     })

    # df = pd.DataFrame(rows)
    # output_csv = Path("./results/entity-linking/huggingface/huggingface.csv")
    # output_csv.parent.mkdir(parents=True, exist_ok=True)
    # df.to_csv(output_csv, index=False)
    # print(f"Saved {len(df)} canonical dataset entries to canonical_datasets_hf.csv")

    # Example: canonical list from Papers with Code
    # huggingface
    # huggingface_entity_path = Path("./results/entity-linking/huggingface/huggingface.csv")
    # huggingface_df = pd.read_csv(huggingface_entity_path)
    # canonical_entities = huggingface_df['name'].tolist()


    zenodo_entity_path = Path("./results/entity-linking/zenodo/zenodo.csv")
    zenodo_df = pd.read_csv(zenodo_entity_path)
    zenodo_df = zenodo_df[zenodo_df.type == 'Dataset']
    logging.info(f"zenodo_df: dataset: {len(zenodo_df)}")
    # canonical_entities = zenodo_df['gold_name'].tolist()
    canonical_entities = zenodo_df['name'].tolist()
    THRESH = 0.7  # your current threshold on RapidFuzz (0..100). Tune on a dev set.
    TOPK = args.topk     # for optional Hits@k/MRR
    BATCH = 256
    # ----------------------------
    # 2) Predict best match (top-1), keep top-k for ranking metrics
    # ----------------------------
    if args.method == 'keyword':
        pred_df = keyword_matching(detected_entities, canonical_entities, TOPK=args.topk, THRESH=THRESH)
    elif args.method == 'semantic':
        pred_df = semantic_matching(detected_entities, canonical_entities, TOPK=args.topk, THRESH=THRESH)
    # ----------------------------
    # 4) Metrics
    # ----------------------------
    def safe_div(a, b):
        return a / b if b else 0.0


    eval_df = nerdme_df[['entity_text','gold_name']].merge(pred_df, on='entity_text', how='left')
    eval_df['gold_is_nil'] = (eval_df['gold_name'].astype(str).str.upper() == 'NIL')
    eval_df['pred_is_nil'] = (eval_df['pred_name'].astype(str).str.upper() == 'NIL')
    eval_df['is_correct_link'] = (~eval_df['gold_is_nil']) & (~eval_df['pred_is_nil']) & (eval_df['gold_name'] == eval_df['pred_name'])

    # (A) Disambiguation (only mentions with gold != NIL)
    disamb = eval_df[~eval_df['gold_is_nil']]
    gold_links = len(disamb)
    pred_links_on_linkable = (~disamb['pred_is_nil']).sum()
    correct_links = disamb['is_correct_link'].sum()

    disamb_precision = safe_div(correct_links, pred_links_on_linkable)       # = correct_links / predicted_links
    disamb_recall    = safe_div(correct_links, gold_links)                   # = correct_links / gold_links
    disamb_f1        = safe_div(2*disamb_precision*disamb_recall, (disamb_precision + disamb_recall))

    # (B) End-to-end with NIL classification
    TP_nil = ((eval_df['gold_is_nil']) & (eval_df['pred_is_nil'])).sum()
    FP_nil = ((eval_df['gold_is_nil']) & (~eval_df['pred_is_nil'])).sum()    # over-linking
    FN_nil = ((~eval_df['gold_is_nil']) & (eval_df['pred_is_nil'])).sum()    # under-linking

    nil_precision = safe_div(TP_nil, TP_nil + FP_nil)
    nil_recall    = safe_div(TP_nil, TP_nil + FN_nil)
    nil_f1        = safe_div(2*nil_precision*nil_recall, (nil_precision + nil_recall))

    # Micro P/R/F1 over link decisions (treat NIL as "no link")
    pred_links_all   = (~eval_df['pred_is_nil']).sum()
    gold_links_all   = (~eval_df['gold_is_nil']).sum()
    correct_links_all= eval_df['is_correct_link'].sum()

    micro_p = safe_div(correct_links_all, pred_links_all)
    micro_r = safe_div(correct_links_all, gold_links_all)
    micro_f1= safe_div(2*micro_p*micro_r, (micro_p + micro_r))

    # ----------------------------
    # 5) Optional: Hits@k and MRR on linkable mentions using ranked candidates
    # ----------------------------
    def hits_at_k(row, k=1):
        if row['gold_is_nil']:
            return None
        cands = row['candidates'] if isinstance(row['candidates'], list) else []
        return int(row['gold_name'] in cands[:k])

    def mrr_row(row):
        if row['gold_is_nil']:
            return None
        cands = row['candidates'] if isinstance(row['candidates'], list) else []
        if row['gold_name'] in cands:
            return 1.0 / (cands.index(row['gold_name']) + 1)
        return 0.0

    # Compute only if you provided gold names (not NIL)
    if (~eval_df['gold_is_nil']).any():
        hits1 = eval_df.apply(hits_at_k, axis=1, k=1).dropna().mean()
        hits3 = eval_df.apply(hits_at_k, axis=1, k=3).dropna().mean()
        hits5 = eval_df.apply(hits_at_k, axis=1, k=5).dropna().mean()
        mrr   = eval_df.apply(mrr_row, axis=1).dropna().mean()
    else:
        hits1 = hits3 = hits5 = mrr = None

    # ----------------------------
    # 6) Print summary
    # ----------------------------
    print("\n=== Entity Linking Evaluation ===")
    print(f"Threshold: {THRESH*100}")
    print(f"Total mentions: {len(eval_df)} | Linkable (gold!=NIL): {gold_links_all} | Predicted links: {pred_links_all}")
    print(f"[Disambiguation]  P={disamb_precision:.4f}  R={disamb_recall:.4f}  F1={disamb_f1:.4f}")
    print(f"[NIL Handling]    P={nil_precision:.4f}  R={nil_recall:.4f}  F1={nil_f1:.4f}")
    print(f"[Micro overall]   P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f1:.4f}")
    if hits1 is not None:
        print(f"[Ranking]         Hits@1={hits1:.4f}  Hits@3={hits3:.4f} Hits@5={hits5:.4f}  MRR={mrr:.4f}")

    # Optionally save the detailed table for error analysis
    eval_df.to_csv(f"./results/entity-linking/eval_predictions_{target_label}_{args.topk}.csv", index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--target_label", type=str, required=False, default='DATASET')
    parser.add_argument("--topk", type=int, required=False, default=5)
    parser.add_argument("--method", type=str, choices=['keyword', 'semantic'])
    args = parser.parse_args()

    main(args)
