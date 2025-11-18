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
from dataclasses import dataclass

# Third-party
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
import requests
# from huggingface_hub import list_datasets
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
from base_predictor import LABELS
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Config:
    threshold: float = 0.7
    topk: int = 10
    model = None


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

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

def fetch_zenodo_records(query, max_records = 200):
    """Fetch a few hundred records from Zenodo for canonical linking."""
    url = "https://zenodo.org/api/records"
    page = 1
    all_items = []
    # while len(all_items) < max_records:
    r = requests.get(url, params={"q": query, "size": 10, "page": page})
    hits = r.json().get("hits", {}).get("hits", [])

    for h in hits:
        md = h.get("metadata", {})
        title = md.get("title", "")
        doi = md.get("doi", "")
        resource_type = md.get("resource_type", {}).get("type", "")
        import ipdb; ipdb.set_trace()
        all_items.append({
            "canonical_id": str(h.get("id", "")),
            "name": title.strip(),
            "gold_name": query,
            "type": resource_type.lower(),
            "aliases": title.lower(),
            "homepage": h.get("links", {}).get("html", ""),
            "doi": doi,
            "source": "Zenodo"
        })

    # page += 1
    return all_items

    # # datasets = list_datasets(limit=1000)
    # # rows = []
    # # for ds in datasets:
    # #     rows.append({
    # #         "canonical_id": ds.id.lower().replace("/", "_"),
    # #         "name": ds.id.split("/")[-1],
    # #         "type": "Dataset",
    # #         "aliases": ds.id,
    # #         "homepage": f"https://huggingface.co/datasets/{ds.id}",
    # #         "source": "HuggingFace Hub"
    # #     })

    # # df = pd.DataFrame(rows)
    # # output_csv = Path("./results/entity-linking/huggingface/huggingface.csv")
    # # output_csv.parent.mkdir(parents=True, exist_ok=True)
    # # df.to_csv(output_csv, index=False)
    # # logging.info(f"Saved {len(df)} canonical dataset entries to canonical_datasets_hf.csv")

    # # Example: canonical list from Papers with Code
    # # huggingface
    # # huggingface_entity_path = Path("./results/entity-linking/huggingface/huggingface.csv")
    # # huggingface_df = pd.read_csv(huggingface_entity_path)
    # # canonical_entities = huggingface_df['name'].tolist()


def keyword_matching(nerdme_df: pd.DataFrame, canonical_entities: List[str], *, cfg: Config = Config()):
    from rapidfuzz import process, fuzz

    results = []
    for i, row in tqdm(nerdme_df.iterrows(), desc="Linking (keyword)"):
        ent = row['entity_text']
        # top-1
        best = process.extractOne(ent, canonical_entities, scorer=fuzz.token_sort_ratio)
        # top-k (for Hits@k/MRR). Returns list of tuples (match, score, idx)
        topk = process.extract(ent, canonical_entities, scorer=fuzz.token_sort_ratio, limit=cfg.topk)
        if best is None:
            pred_name, score = 'NIL', 0
            topk_names = []
        else:
            pred_name, score, _ = best
            if score < cfg.threshold * 100:
                pred_name = 'NIL'
            topk_names = [m[0] for m in topk] if topk else []

        results.append({
            'entity_text': ent,
            'pred_name': pred_name,
            'score': score,
            'candidates': topk_names,
            'gold_name': row['gold_name'],
            'entity_type': row['entity_type'],
        })
    df = pd.DataFrame(results)
    return df

def semantic_matching(nerdme_df: pd.DataFrame, canonical_entities: List[str], *, cfg: Config = Config()):
    import torch
    import sentence_transformers as st

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 2) Pre-encode canonical entities once (normalized embeddings)
    if not cfg.model:
        model = st.SentenceTransformer('all-MiniLM-L6-v2')
        model = model.to(device)
    else:
        model = cfg.model

    detected_entities = nerdme_df['entity_text'].tolist()
    detected_emb = model.encode(detected_entities, convert_to_tensor=True)
    canonical_emb = model.encode(canonical_entities, convert_to_tensor=True)
    cosine_scores = st.util.cos_sim(detected_emb, canonical_emb).cpu()
    results = []
    for i, row in tqdm(nerdme_df.iterrows(), desc='Linking (semantic)'):
        ent = row['entity_text']
        scores = cosine_scores[i]
        topk_idx = scores.argsort().numpy()[::-1][:cfg.topk]
        best_idx = topk_idx[0]
        best_score = scores[best_idx].item()

        topk_names = [canonical_entities[j] for j in topk_idx]
        pred_name = canonical_entities[best_idx] if best_score >= cfg.threshold else 'NIL'
        results.append({
            "entity_text": ent,
            "pred_name": pred_name,
            "score": best_score,     # cosine in [-1,1]; typically 0.0–0.9 for good matches
            "candidates": topk_names,
            'gold_name': row['gold_name'],
            'entity_type': row['entity_type'],
        })

    df = pd.DataFrame(results)
    return df

def finetune_matching(nerdme_df_train: pd.DataFrame, nerdme_df: pd.DataFrame, canonical_entities: List[str], *, cfg: Config = Config()):
    import torch
    from torch.utils.data import DataLoader
    # from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
    import sentence_transformers as st

    gold_col = 'gold_name'
    nerdme_df_train["entity_text"] = nerdme_df_train["entity_text"].astype(str).str.lower().str.strip()
    nerdme_df_train[gold_col] = nerdme_df_train[gold_col].astype(str).str.lower().str.strip()

    train_examples = []
    # Build input pairs (mention/context) → (canonical PwC name)
    for _, row in nerdme_df_train.iterrows():
        mention = row["entity_text"]
        context = row.get("sentence", "")
        text_left = mention if not context else f"{mention} [SEP] {context}"
        text_right = row[gold_col]
        train_examples.append(st.InputExample(texts=[text_left, text_right]))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = st.SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to(device)

    train_dataloader = DataLoader(train_examples, batch_size=32)
    train_loss = st.losses.MultipleNegativesRankingLoss(model)

    num_epochs = 10
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    model_path = Path(f"./finetuned_minilm_nerdme_pwc_{num_epochs}")
    # ==================================================
    # 4. Train
    # ==================================================
    if not model_path.exists():
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=str(model_path)
        )
        logging.info(f"Model saved to {model_path}")
    model = st.SentenceTransformer(str(model_path))
    cfg.model = model
    return semantic_matching(nerdme_df, canonical_entities, cfg=cfg)


def load_pwc_dataset():
    """
    Loads the Papers with Code dataset catalog from HuggingFace.
    Expects a dataset with a 'train' split and fields like 'name', 'full_name', 'homepage'.
    """
    pwc = load_dataset("pwc-archive/datasets")

    df = pwc["train"].to_pandas()

    # Build a canonical name field. You can tune this.
    # Here we prefer verified_name or name/full_name if available.
    def canonical_name(row):
        if isinstance(row.get("verified_name"), str) and row["verified_name"].strip():
            return row["verified_name"].strip().lower()
        # if isinstance(row.get("name"), str) and row["name"].strip():
        #     return row["name"].strip()
        # if isinstance(row.get("full_name"), str) and row["full_name"].strip():
        #     return row["full_name"].strip()
        return None

    df["canonical_name"] = df.apply(canonical_name, axis=1)
    df = df.dropna(subset=["canonical_name"]).reset_index(drop=True)
    # Add a simple canonical id (row index) if you do not want to use URLs
    df["canonical_id"] = df.index.astype(int)
    return df

def load_zenodo_dataset(nerdme_df: pd.DataFrame, target_label: str):
    detected_entities = nerdme_df[nerdme_df['gold_name'] != 'NIL']['gold_name'].unique().tolist()
    entities_to_query = []
    specials = ['-', '|', '_', '+', '=', '>', '<']
    for entity in detected_entities:
        entity = entity.lower()
        if '(' in entity:
            entity = entity.replace('(', '')
        if ')' in entity:
            entity = entity.replace(')', '')

        if '/' in entity:
            entity = entity.split('/')[-1]

        for sp in specials:
            if sp in entity:
                entity = entity.replace(sp, ' ')

        if entity in entities_to_query:
            continue
        entity = normalize_spaces(entity)
        entities_to_query.append(entity)
    logging.info(f"NERDME unique entities: {len(entities_to_query)}")

    all_items = []
    save_path = Path(f"results/entity-linking/zenodo/{target_label}.csv")
    ckpt_path = Path(f'results/entity-linking/zenodo/ckpt_{target_label.lower()}.txt')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists():
        saved_idx = int(ckpt_path.read_text())
        df = pd.read_csv(save_path)
        for i, row in df.iterrows():
            all_items.append(dict(row))
    else:
        saved_idx = 0

    last_saved_item_cnt = 0

    for idx, query in enumerate(entities_to_query):
        if idx <= saved_idx:
            continue

        items = fetch_zenodo_records(query, max_records=10)
        if len(items) == 0:
            logging.info(f"[{idx:5d}]-[{query}] has no corresponding {target_label} on Zenodo")

        all_items.extend(items)
        df = pd.DataFrame(all_items)
        df.to_csv(save_path, index=False)
        logging.info(f"[{idx:5d}]-[{query}] saved {len(all_items) - last_saved_item_cnt} Zenodo canonical entries to {save_path}")
        ckpt_path.write_text(str(idx))
        last_saved_item_cnt = len(all_items)
        time.sleep(0.3)

    df = pd.DataFrame(all_items)
    import ipdb; ipdb.set_trace()
    return df

def load_nerdme_mentions(file: Path):
    """
    Expects a CSV with at least:
        - entity_text  (the dataset mention in NERdME)
    Optionally:
        - gold_pwc_name or gold_pwc_id (for evaluation, if you have it)
    """
    df = pd.read_csv(path)
    # Basic cleaning
    df["entity_text"] = df["entity_text"].astype(str).str.strip()
    return df

def evaluate_linking(df: pd.DataFrame, gold_col: str = 'gold_name'):
    """
    pred_df: output of keyword_matching or semantic_matching
    gold_df: NERdME mentions with a gold column specifying the correct PwC name
    gold_col: column in gold_df that stores the gold canonical name (string)
    """
    # Classification metrics
    tp = ((df["pred_name"] == df[gold_col]) & (df[gold_col] != "NIL")).sum()
    fp = ((df["pred_name"] != df[gold_col]) & (df["pred_name"] != "NIL")).sum()
    fn = ((df["pred_name"] == "NIL") & (df[gold_col] != "NIL")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Ranking metrics (Hits@k and MRR)
    def hits_at_k(row, k):
        gold = row[gold_col]
        if gold == "NIL":
            return 0.0
        return 1.0 if gold in row["candidates"][:k] else 0.0

    def reciprocal_rank(row):
        gold = row[gold_col]
        if gold == "NIL":
            return 0.0
        cand = row["candidates"]
        if gold in cand:
            r = cand.index(gold) + 1
            return 1.0 / r
        return 0.0

    hits1 = df.apply(lambda r: hits_at_k(r, 1), axis=1).mean()
    hits3 = df.apply(lambda r: hits_at_k(r, 3), axis=1).mean()
    mrr = df.apply(reciprocal_rank, axis=1).mean()

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "hits@1": hits1 * 100,
        "hits@3": hits3 * 100,
        "mrr": mrr * 100,
    }


def main(args):
    # Prepare NERdME entity
    platform = 'zenodo'
    files = [x for x in Path("data/train").rglob("*.tsv")] + [x for x in Path("data/val").rglob("*.tsv")] + [x for x in Path("data/test_labeled/").rglob("*.tsv")]
    for target_label in LABELS:
        save_path = Path(f'results/entity-linking/nerdme/{target_label}.csv')
        if save_path.exists() and save_path.stat().st_size > 0:
            continue
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df = extract_entity_from_tsv(files, target_label=target_label)
        logging.info(f"✅ Extracted {len(df)} {target_label} entities → {save_path}")
        df.to_csv(save_path, index=False)

    gold_nerdme_path = Path(f'results/entity-linking/nerdme/{args.target_label}_GOLD_{platform}.csv')
    # Attach gold if available; otherwise default NIL
    if gold_nerdme_path.exists():
        # pure_nerdme_path = Path(f'results/entity-linking/nerdme/{args.target_label}.csv')
        # pure_nerdme_df = pd.read_csv(pure_nerdme_path)
        # nerdme_df['gold_name'] = nerdme_df['entity_text'].apply(lambda x: x.lower())
        # nerdme_df.to_csv(gold_nerdme_path, index=False)
        nerdme_df = pd.read_csv(gold_nerdme_path)
        nerdme_df['gold_name'] = nerdme_df[f'{platform}_gold_name']
        # nerdme_df_merged = nerdme_df.merge(pure_nerdme_df, on='entity_text')
        # nerdme_df_merged.to_csv(gold_nerdme_path, index=False)
    else:
        raise Exception(f"[{args.target_label}] no curated gold name: {gold_nerdme_path}")

    nerdme_df_shuffled = nerdme_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_frac = 0.8  # 80% / 20% split (adjust as needed)
    split_idx = int(len(nerdme_df_shuffled) * train_frac)

    nerdme_df_train = nerdme_df_shuffled.iloc[:split_idx]
    nerdme_df = nerdme_df_shuffled.iloc[split_idx:]
    nerdme_df_train = nerdme_df_train.reset_index(drop=True)
    nerdme_df = nerdme_df.reset_index(drop=True)
    logging.info(f'Train: {len(nerdme_df_train)}, Test: {len(nerdme_df)}')
    if platform == 'pwc':
        pwc_df = load_pwc_dataset()
        canonical_entities = pwc_df["canonical_name"].str.lower().tolist()
    elif platform == 'zenodo':
        zenodo_df = load_zenodo_dataset(nerdme_df, args.target_label)
        import ipdb; ipdb.set_trace()
    # 3) Run keyword based linking
    kw_pred = keyword_matching(nerdme_df, canonical_entities)
    kw_result = evaluate_linking(kw_pred)
    logging.info(f"Keyword result:\n{json.dumps(kw_result, indent=2)}")
    # 4) Run semantic linking
    sem_pred = semantic_matching(nerdme_df, canonical_entities)
    sem_result = evaluate_linking(sem_pred)
    logging.info(f"Semantic result:\n{json.dumps(sem_result, indent=2)}")
    # 5) Run finetuned semantic linking
    ft_pred = finetune_matching(nerdme_df_train, nerdme_df, canonical_entities)
    ft_result = evaluate_linking(ft_pred)
    logging.info(f"Finetuned result:\n{json.dumps(ft_result, indent=2)}")

    overall_result = {
        'keyword': kw_result,
        'semantic': sem_result,
        'finetine': ft_result,
    }
    save_path = Path(f'results/entity-linking/{platform}_{args.target_label.lower()}.json')
    save_path.write_text(json.dumps(overall_result, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--target_label", type=str, required=False, default='DATASET')
    parser.add_argument("--topk", type=int, required=False, default=5)
    # parser.add_argument("--method", type=str, choices=['keyword', 'semantic'])
    parser.add_argument("--platform", type=str, choices=['pwc', 'zenodo'])
    args = parser.parse_args()

    main(args)
