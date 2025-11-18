# Standard library
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


from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
import numpy as np

from collections import defaultdict, Counter
from src.webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



train_data_dir = Path("./data/train")
test_data_dir = Path("./data/test_labeled")
val_data_dir = Path("./data/val")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_annotations(files):
    annotations = []
    for file_path in files:
        doc = webanno_tsv_read_file(file_path)
        annotations.extend([x for x in doc.annotations])

    print(f'{len(annotations)} annotations found in {len(files)} files.')
    label_to_entities = defaultdict(list)
    for annotation in annotations:
        label_to_entities[annotation.label].append(annotation.text)

    total_span, total_unique_span = 0, 0
    label_to_entities = dict(sorted(label_to_entities.items()))
    for label, entities in label_to_entities.items():
        total_span += len(entities)
        print(f'{label}:\t{len(entities)}')
    print('======================')
    for label, entities in label_to_entities.items():
        unique_entities = set(entities)
        total_unique_span += len(unique_entities)
        print(f'{label}:\t{len(unique_entities)}')

    return label_to_entities, total_span, total_unique_span

def perplexity(text: str, tokenizer: Any, model: Any) -> float:
    if not text.strip():
        return float("nan")
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # average over tokens

    ppl = torch.exp(loss).item()
    return ppl


def main(args):
    # Load a pretrained LM (you can switch to 'gpt2' if you like)
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.eval()

    model.to(device)

    results = []
    train_files = [x for x in train_data_dir.rglob("*.tsv")]
    val_files  = [x for x in val_data_dir.rglob("*.tsv")]
    test_files = [x for x in test_data_dir.rglob("*.tsv")]
    all_files = train_files + val_files + test_files
    label_to_entities, _, _ = get_annotations(all_files)

    groups = {
        "Paper": ["CONFERENCE", "WORKSHOP", "PUBLICATION", "EVALMETRIC",  "DATASET"],
        "Implementation": ["SOFTWARE", "PROGLANG", "LICENSE"],
        "Others": ["PROJECT", "ONTOLOGY"],
    }

    group_to_entities = defaultdict(list)
    for label, entities in label_to_entities.items():
        print(f'Processing label: {label} with {len(entities)} entities, unique entities is {len(set(entities))}.')
        for group_name, group_labels in groups.items():
            if label in group_labels:
                group_to_entities[group_name].extend(entities)

    # Compute perplexity per mention
    group_to_ppls = defaultdict(list)
    # Compute per-group perplexity
    for group, mentions in group_to_entities.items():
        ppls = []
        for m in mentions:
            ppl = perplexity(m, tokenizer, model)
            if not np.isnan(ppl):
                ppls.append(ppl)
        ppls = np.array(ppls)
        print(f"{group:<15} | n={len(ppls):4d} | meanPPL={ppls.mean():.2f} | std={ppls.std():.2f}")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    main(args)
