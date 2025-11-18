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
# import evaluate
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, get_cosine_schedule_with_warmup, AutoTokenizer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report
from sklearn.metrics import accuracy_score

from base_predictor import BasePredictor, LABELS
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
from compute_metrics import compute_metrics_exact, compute_metrics_partial, flatten

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


label2id = None
id2label = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tokenize_and_preserve_labels(tokens, labels, tokenizer):
    tokenized_tokens = []
    tokenized_labels = []
    for token, label in zip(tokens, labels):
        tokenized_word = tokenizer.tokenize(token)
        n_subwords = len(tokenized_word)
        # Add the tokenized word to the final tokenized word list
        tokenized_tokens.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        tokenized_labels.extend([label] * n_subwords)

    return tokenized_tokens, tokenized_labels


class dataset(Dataset):
    def __init__(self, all_tokens, all_labels, tokenizer, max_len):
        self.all_tokens = all_tokens
        self.all_labels = all_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Build word-level dataset
    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        tokens = self.all_tokens[index]
        labels = self.all_labels[index]

        tokenized_tokens, tokenized_labels = tokenize_and_preserve_labels(tokens, labels, self.tokenizer)
        # step 2: add special tokens (and corresponding labels)
        tokenized_tokens = [self.tokenizer.cls_token] + tokenized_tokens + [self.tokenizer.sep_token] # add special tokens of Roberta
        tokenized_labels.insert(0, "O") # add outside label for [CLS] token
        tokenized_labels.append("O") # add outside label for [SEP] token
        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_tokens) > maxlen):
            # truncate
            tokenized_tokens = tokenized_tokens[:maxlen]
            tokenized_labels = tokenized_labels[:maxlen]
        else:
            # pad
            tokenized_tokens = tokenized_tokens + [self.tokenizer.pad_token for _ in range(maxlen - len(tokenized_tokens))]
            tokenized_labels = tokenized_labels + ["O" for _ in range(maxlen - len(tokenized_labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != self.tokenizer.pad_token else 0 for tok in tokenized_tokens] #modifié selon https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/camembert

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_tokens)
        label_ids = [label2id[label] for label in tokenized_labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long),
        }

    def __len__(self):
        return len(self.all_tokens)

    def sentence(self, idx):
        return ' '.join(self.all_tokens[idx])

    def label(self, idx):
        return ' '.join(self.all_labels[idx])


def build_dataset_word(folder: Path, tokenizer: Any, target_label: str):
    all_tokens = []
    all_labels = []
    file_paths = sorted([x for x in folder.rglob('*.tsv')])

    for file in file_paths:
        tokens = []
        labels = []
        for line in file.read_text().split('\n'):
            line = line.strip()
            if line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 5:
                continue

            _, _, token, _, label = parts
            if label in LABELS:
                label = f'I-{label}'
            else:
                label = 'O'

            label = label.split('[')[0]
            tokens.append(token)
            labels.append(label)

        for idx in range(len(labels)-1):
            if labels[idx] == 'O' and labels[idx+1].startswith('I-'):
                labels[idx+1] = labels[idx+1].replace('I-', 'B-')

        all_tokens.append(tokens)
        all_labels.append(labels)

    ds = dataset(all_tokens, all_labels, tokenizer, max_len=256)
    return ds

def build_dataset_char(folder: Path, tokenizer: Any, target_label: str):
    file_paths = sorted([x for x in folder.rglob('*.tsv')])

    all_tokens = []
    all_labels = []
    max_len = 0
    for file in file_paths:
        doc = webanno_tsv_read_file(file)
        anno_tokens = []
        for annotation in doc.annotations:
            if annotation.label != target_label:
                continue

            anno_tokens += annotation.tokens
        if len(anno_tokens) == 0:
            continue

        for sentence in doc.sentences:
            labels = []
            tokens = []
            sent_tokens = doc.sentence_tokens(sentence)
            annotated = False
            for token in sent_tokens:
                tokens.append(token.text)
                if token in anno_tokens:
                    labels.append(f'I-{target_label}')
                    annotated = True
                else:
                    labels.append('O')

            if len(tokens) != len(labels):
                raise Exception("tokens and labels are mismatched")

            for idx in range(len(labels) - 1):
                if labels[idx] == 'O' and labels[idx+1] == f'I-{target_label}':
                    labels[idx+1] = f'B-{target_label}'

            if annotated is True:
                all_tokens.append(tokens)
                all_labels.append(labels)

            # for token, label in zip(all_tokens[0][:30], all_labels[0][:30]):
            #     logging.info('{0:15}  {1}'.format(token, label))

    # logging.info('------------------------------------------------------------')
    # for token, label in zip(all_tokens[0][:50], all_labels[0][:50]):
    #     logging.info('{0:15}  {1}'.format(token, label))
    ds = dataset(all_tokens, all_labels, tokenizer, max_len=256)
    # logging.info the first 50 tokens and corresponding labels
    # logging.info('------------------------------------------------------------')
    # for token, label in zip(all_tokens[0][:50], all_labels[0][:50]):
    #     logging.info('{0:15}  {1}'.format(token, label))

    # for token, label in zip(tokenizer.convert_ids_to_tokens(ds[0]["ids"][:50]), ds[0]["targets"][:50]):
    #     logging.info('{0:15}  {1}'.format(token, id2label[label.item()]))
    # logging.info('------------------------------------------------------------')
    # for token, label in zip(all_tokens[0][:50], all_labels[0][:50]):
    #     logging.info('{0:15}  {1}'.format(token, label))

    return ds

# Defining the training function on the 80% of the dataset for tuning the bert model
def train(model, train_loader, optimizer, scheduler=None):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    for idx, batch in enumerate(train_loader):

        ids = batch['ids'].to(device, dtype = torch.long)
        mask = batch['mask'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs.loss, outputs.logits
        ''' loss, tr_logits  = model(input_ids=ids, attention_mask=mask, labels=targets)#temporary modification for transformer 3'''

        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        #if idx % 100==0:
        #    loss_step = tr_loss/nb_tr_steps
        #    logging.info(f"Training loss per 100 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        # import ipdb; ipdb.set_trace()
        tr_preds.extend(predictions)
        tr_labels.extend(targets)
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
        MAX_GRAD_NORM = 10
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    #logging.info(f"Trained {nb_tr_steps} steps")
    logging.info(f"Training loss epoch: {epoch_loss}")
    logging.info(f"Training accuracy epoch: {tr_accuracy}")


def valid(model, validation_loader, tokenizer, target_label):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    eval_tokens = []
    all_char_level_preds = []
    all_char_level_labels = []
    with torch.no_grad():
        for idx, batch in enumerate(validation_loader):
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)

            ids = ids.view(1, -1)
            mask = mask.view(1, -1)
            targets = targets.view(1, -1)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            #if idx % 100==0:
            #    loss_step = eval_loss/nb_eval_steps
            #    logging.info(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(targets)
            eval_preds.extend(predictions)
            tokenized_tokens = tokenizer.convert_ids_to_tokens(torch.masked_select(ids, active_accuracy).cpu())
            tokenized_labels = [id2label[label.item()] for label in targets.cpu()]
            tokenized_preds = [id2label[label.item()] for label in predictions.cpu()]
            tokenized_tokens = tokenized_tokens[1:-1]
            tokenized_labels = tokenized_labels[1:-1]
            tokenized_preds  = tokenized_preds[1:-1]
            char_level_preds = []
            char_level_labels = []
            for token, pred, label in zip(tokenized_tokens, tokenized_preds, tokenized_labels):
                if target_label in pred:
                    char_level_preds.extend([f'I-{target_label}'] * len(token))
                else:
                    char_level_preds.extend(['O'] * len(token))

                if target_label in label:
                    char_level_labels.extend([f'I-{target_label}'] * len(token))
                else:
                    char_level_labels.extend(['O'] * len(token))

            for idx in range(1, len(char_level_preds)):
                if char_level_preds[idx-1] == 'O' and char_level_preds[idx] == f'I-{target_label}':
                    char_level_preds[idx] = f'B-{target_label}'

                if char_level_labels[idx-1] == 'O' and char_level_labels[idx] == f'I-{target_label}':
                    char_level_labels[idx] = f'B-{target_label}'

            if char_level_preds[0] == f'I-{target_label}':
                char_level_preds[0] = f'B-{target_label}'

            if char_level_labels[0] == f'I-{target_label}':
                char_level_labels[0] = f'B-{target_label}'

            all_char_level_preds.append(char_level_preds)
            all_char_level_labels.append(char_level_labels)

            eval_tokens.extend(tokenized_tokens)
            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    #logging.info(eval_labels)
    #logging.info(eval_preds)

    # x, y = flatten(all_char_level_labels), flatten(all_char_level_preds)
    # for idx in range(1, len(x)):
    #     if x[idx-1] == 'O' and x[idx] == f'I-{target_label}':
    #         import ipdb; ipdb.set_trace()

    # for idx in range(1, len(y)):
    #     if y[idx-1] == 'O' and y[idx] == f'I-{target_label}':
    #         import ipdb; ipdb.set_trace()

    labels = [id2label[id.item()] for id in eval_labels]
    predictions = [id2label[id.item()] for id in eval_preds]
    #logging.info(labels)
    #logging.info(predictions)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    logging.info(f"Validation Loss: {eval_loss}")
    logging.info(f"Validation Accuracy: {eval_accuracy}")
    # ret = compute_metrics_exact(all_char_level_labels, all_char_level_preds, target_label)
    # logging.info(f"Exact: {ret}")
    ret = compute_metrics_partial(all_char_level_labels, all_char_level_preds, target_label)
    logging.info(f"{target_label} Partial: {ret}")
    ret = compute_metrics_exact(all_char_level_labels, all_char_level_preds, target_label)
    logging.info(f"{target_label} Exact: {ret}")
    # ret = classification_report([labels], [predictions], mode='strict', output_dict=True)
    # for k, v in ret.items():
    #     for kk, vv in v.items():
    #         ret[k][kk] = float(vv)

    # logging.info(f"{target_label} Exact: {ret}")
    # return labels, predictions
    return flatten(all_char_level_labels), flatten(all_char_level_preds)
    # return all_char_level_labels, all_char_level_preds


def print_reports_to_csv(test_results, model_name, LEARNING_RATE, EPOCHS, report_type):
    import json
    test_reports = []
    for res in test_results:
        # report1 = classification_report([res['labels']], [res['predictions']], mode='partial', output_dict=True)
        report1 = compute_metrics_partial(res['labels'], res['predictions'], '')
        report2 = compute_metrics_exact(res['labels'], res['predictions'], '')

        report3 = classification_report([res['labels']], [res['predictions']], mode='strict', output_dict=True)
        for k, v in report3.items():
            for kk, vv in v.items():
                report3[k][kk] = float(vv)

        test_reports.append({"partial": report1, "exact": report2, "exact2": report3})

    test_report_name = Path(f'./results/finetuning_results/{report_type}_{model_name}_{LEARNING_RATE}_16_{EPOCHS}.json')
    test_report_name.parent.mkdir(parents=True, exist_ok=True)
    test_report_name.write_text(json.dumps(test_reports, sort_keys=True, indent=2))
    logging.info(f"save results to path: {test_report_name}")


def main(args):
    label_set = []
    for label in [args.target_label]:
        label_set.append(f'B-{label}')
        label_set.append(f'I-{label}')
    label_set.append('O')
    global label2id
    global id2label
    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}

    MAX_LEN = 256
    batch_size = 16
    train_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }

    train_folder = Path("./data/train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, from_tf=False, model_max_length=MAX_LEN)
    # build_dataset = build_dataset_word if args.mode == 'word' else build_dataset_char
    build_dataset = build_dataset_char
    train_set = build_dataset(Path("data/train"), tokenizer, args.target_label)
    valid_set = build_dataset(Path("data/val"), tokenizer, args.target_label)
    test_set = build_dataset(Path("data/test_labeled"), tokenizer, args.target_label)
    train_loader = DataLoader(train_set, **train_params)
    valid_loader = DataLoader(valid_set, **val_params)
    test_loader = DataLoader(test_set, **val_params)


    logging.info("TRAIN Dataset: {}".format(len(train_set)))
    #train_params['batch_size'] =  int( trainsetsize / 32) if (trainsetsize < 1024) else 16
    # EPOCHS = 1 #3#20
    LEARNING_RATE = 5e-5 #1e-05
    train_loader = DataLoader(train_set, **train_params)
    # num_training_steps = int(len(train_loader) / train_params['batch_size'] * EPOCHS)
    # logging.info(f'tranining steps: {num_training_steps+1}')

    #Shrey uses TF model
    ner_model_name = Path(f'ner_model/{args.model}_ft_{args.epoch}_{args.target_label}')
    ner_model_name.parent.mkdir(parents=True, exist_ok=True)
    if (ner_model_name / "model.safetensors").exists():
        tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
    else:
        model = AutoModelForTokenClassification.from_pretrained(args.model,
                                                                from_tf=False,
                                                                num_labels=len(id2label),
                                                                id2label=id2label,
                                                                label2id=label2id)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
        for epoch in range(args.epoch):
            logging.info(f"Training epoch: {epoch + 1}")
            train(model, train_loader, optimizer)
            #valid(model, validation_loader)
            #valid(model, test_gen_loader)

        model.save_pretrained(ner_model_name)
        tokenizer.save_pretrained(ner_model_name)

    model.to(device)
    #scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 50, num_training_steps=num_training_steps)
    # labels, predictions = valid(model, test_loader, tokenizer)
    labels, predictions = valid(model, test_set, tokenizer, target_label=args.target_label)
    # save_path = Path(f"./results/{args.model}/{args.mode}/test_labeled/result_{args.target_label}.json")
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # result = {"truth": labels, "pred": predictions}
    # save_path.write_text(json.dumps(result))

    test_results = []
    test_results.append({ 'model': args.model, 'labels': labels, 'predictions': predictions})
    print_reports_to_csv(test_results, args.model, LEARNING_RATE, args.epoch, f'test-{args.target_label}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument('--mode', type=str, default='word')
    parser.add_argument('--target_label', type=str, default='DATASET')
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    main(args)
