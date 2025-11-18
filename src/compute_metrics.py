import argparse
import json
import os
from collections import defaultdict
from functools import reduce
import pandas as pd
from pathlib import Path
from webanno_tsv import webanno_tsv_read_file, Document, Annotation
from typing import List, Union


LABELS = [
    'CONFERENCE',
    'DATASET',
    'EVALMETRIC',
    'LICENSE',
    'ONTOLOGY',
    'PROGLANG',
    'PROJECT',
    'PUBLICATION',
    'SOFTWARE',
    'WORKSHOP'
]


def flatten(lst):
    return reduce(lambda x, y: x + y, lst)


def to_char_bio(pred_path: str, ref_path: str) -> List[List[str]]:
    ref_doc = webanno_tsv_read_file(ref_path)
    # Parse the WebAnno TSV file
    pred_doc = webanno_tsv_read_file(pred_path)
    # Initialize a list to store character-level BIO tags
    bio_tags_list = []
    for target_label in LABELS:
        bio_tags = ['#'] * len(ref_doc.text)  # Default to '#' for all characters
        # Pick interested sentences and default them to 'O'
        for annotation in ref_doc.annotations:
            label = annotation.label
            if label != target_label:
                continue
            sentences = ref_doc.annotation_sentences(annotation)
            # print('anntation: ', annotation.text)
            # import ipdb; ipdb.set_trace()
            for sentence in sentences:
                tokens = ref_doc.sentence_tokens(sentence)
                start_char, end_char = tokens[0].start, tokens[-1].end
                bio_tags[start_char:end_char] = ['O'] * (end_char-start_char)
                if end_char < len(bio_tags):
                    bio_tags[end_char] = 'O'

        for annotation in pred_doc.annotations:
            label = annotation.label
            if label != target_label:
                continue

            start_token, end_token = annotation.tokens[0], annotation.tokens[-1]
            start_char = start_token.start
            end_char = end_token.end
            # Sanity check
            if ref_doc.text[start_char:end_char] != annotation.text:
                msg = f"ERROR: pred: {pred_path}, annotated '{annotation.text}', text: '{ref_doc.text[start_char:end_char]}'"
                print(msg)

            if 'I-' in bio_tags[start_char]:
                # Overlapping, it's annotated by another annotations, we connect them as one annotations
                # import ipdb; ipdb.set_trace()
                continue
            else:
                if bio_tags[start_char] == 'O':
                    # Assign BIO tags to characters in the entity span
                    bio_tags[start_char] = f'B-{label}'  # Beginning of the entity
                # else:
                #     import ipdb; ipdb.set_trace()
            for i in range(start_char + 1, end_char):
                if bio_tags[i] == 'O':
                    bio_tags[i] = f'I-{label}'  # Inside the entity
            if 'B-' not in bio_tags[start_char] and 'I-' in bio_tags[start_char + 1]:
                import ipdb; ipdb.set_trace()

        # Remove unannotated sentences from bio list.
        # bio_tags = [x for x in filter(lambda x: x != '#', bio_tags)]
        # bio_tags = [x for x in filter(lambda x: x != '#', bio_tags)]
        for i, tag in enumerate(bio_tags):
            if tag == '#':
                bio_tags[i] = 'O'

        bio_tags_list.append(bio_tags)

    return bio_tags_list


def get_spans(tag_list):
    spans = []
    i = 0
    start = end = None
    while i < len(tag_list):
        tag = tag_list[i]
        if 'B-' in tag:
            start = i
            end = i
        elif 'I-' in tag:
            try:
                end += 1
            except:
                import ipdb; ipdb.set_trace()
        elif 'O' == tag:
            if start is not None:
                spans.append((start, end))

            start = None
            end = None

        i += 1
    return spans


def compute_metrics_exact(y_true_, y_pred_, label):
    if any(isinstance(s, list) for s in y_true_):
        y_true = [item for sublist in y_true_ for item in sublist]
        y_pred = [item for sublist in y_pred_ for item in sublist]
    else:
        y_true = y_true_
        y_pred = y_pred_
        # for i, sublist in enumerate(y_true_):
        #     for j in range(len(sublist)-1):
        #         if sublist[j] == 'O' and 'I-' in sublist[j+1]:
        #             print(f"BUG (true): i: {i}, j: {j}, {label}")
        #             import ipdb; ipdb.set_trace()

        # for i, sublist in enumerate(y_pred_):
        #     for j in range(len(sublist)-1):
        #         if sublist[j] == 'O' and 'I-' in sublist[j+1]:
        #             print(f"BUG (pred): i: {i}, j: {j}, {label}")
        #             import ipdb; ipdb.set_trace()


    # x_i = [i for i in range(len(y_true)-1) if y_true[i] == 'O' and 'I-' in y_true[i+1]]
    # y_i = [i for i in range(len(y_pred)-1) if y_pred[i] == 'O' and 'I-' in y_pred[i+1]]
    # if len(x_i) > 0 or len(y_i) > 0:
    #     print(f'BUG, {label}')
    #     import ipdb; ipdb.set_trace()

    spans_true = get_spans(y_true)
    spans_pred = get_spans(y_pred)

    tp_spans = [x for x in spans_pred if x in spans_true]
    TP_ = sum([x[1] - x[0] + 1 for x in tp_spans])

    FP = 0
    FN = 0
    TN = 0
    TP = 0
    i = 0
    tp_span_start = [x[0] for x in tp_spans]
    # given a label, we don't have overlapping entities
    while i < len(y_true):
        if i in tp_span_start:
            if y_true[i] == 'O':
                print('BUG')

            while y_true[i] != 'O':
                if y_pred[i] == 'O':
                    print('BUG')
                    import ipdb; ipdb.set_trace()
                TP += 1
                i += 1
            continue
        elif y_true[i] == 'O':
            if y_pred[i] != 'O':
                FP += 1
            else:
                TN += 1
        elif y_true[i] != 'O':
            # if y_pred[i] == 'O':        # even y_pred[i] == O, we consider it as O since it only partially matches
            FN += 1
        i += 1

    if TP + FN + TN + FP != len(y_true):
        import ipdb; ipdb.set_trace()

    if TP != TP_:
        import ipdb; ipdb.set_trace()

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_metrics_partial(y_true, y_pred, label):
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    TP = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred) if ((y_t != 'O') or (y_p != 'O')))
    FP = sum(((y_t != y_p) and (y_p != 'O')) for y_t, y_p in zip(y_true, y_pred))
    FN = sum(((y_t != 'O') and (y_p == 'O')) for y_t, y_p in zip(y_true, y_pred))
    TN = sum((y_t == y_p == 'O') for y_t, y_p in zip(y_true, y_pred))

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_macro_micro_weighted_metrics(label_to_metrics):
    total_labels = len(label_to_metrics)
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    label_support = []  # For weighted averaging (label support)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for label, metrics in label_to_metrics.items():
        TP = metrics['TP']
        FN = metrics['FN']
        FP = metrics['FP']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        label_support.append(TP + FN)  # Support is the total number of actual instances of the label

        total_tp += TP
        total_fp += FP
        total_fn += FN

    macro_precision = sum(all_precisions) / total_labels
    macro_recall = sum(all_recalls) / total_labels
    macro_f1 = sum(all_f1_scores) / total_labels

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Weighted average (weighted by label support)
    weighted_precision = sum([all_precisions[i] * label_support[i] for i in range(total_labels)]) / sum(label_support)
    weighted_recall = sum([all_recalls[i] * label_support[i] for i in range(total_labels)]) / sum(label_support)
    weighted_f1 = sum([all_f1_scores[i] * label_support[i] for i in range(total_labels)]) / sum(label_support)

    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def get_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--reference_dir', type=str, help='Path to the reference data, e.g. training/validation/test data', required=True)
    parser.add_argument('--prediction_dir', type=str, help='Path to save the prediction results', required=True)
    parser.add_argument('--mode', type=str, help='partial,exact', required=True)
    return parser



if __name__ == "__main__":
    parser = get_parse()
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir)
    pred_dir = Path(args.prediction_dir)
    mode = args.mode

    ref_file_names = sorted([x for x in ref_dir.rglob('*.tsv')
                             if 'daijifeng' not in str(x) and 'ai4finance' not in str(x)
                             and 'andi611' not in str(x)
                             ])
    for i, f in enumerate(ref_file_names):
        print(f"{i} -> {str(f)}")

    if len(ref_file_names) == 0:
        raise Exception("ERROR: No reference files found, configuration error?")

    all_ref_bio_tags_list = [to_char_bio(ref_path, ref_path) for ref_path in ref_file_names]

    all_pred_bio_tags_list = []
    for idx, ref_path in enumerate(ref_file_names):
        try:
            pred_path = pred_dir / ref_path.name
            all_pred_bio_tags_list.append(to_char_bio(pred_path, ref_path))
        except FileNotFoundError:
            nbr_labels = len(all_ref_bio_tags_list[idx])
            assert nbr_labels == len(LABELS), "ERROR: reference tags doesn't have ${len(LABELS)} labels."
            pred = []
            for label_idx in range(nbr_labels):
                pred.append(['O'] * len(all_ref_bio_tags_list[idx][label_idx]))

            print(f"WARN: {ref_path.name} is missing, fill 'O' list as default prediction")
            all_pred_bio_tags_list.append(pred)

    # Sanity checking
    for idx, (ref_list, pred_list) in enumerate(zip(all_ref_bio_tags_list, all_pred_bio_tags_list)):
        for label_idx, (ref, pred) in enumerate(zip(ref_list, pred_list)):
            assert len(ref) == len(pred), f'ERROR: {ref_file_names[idx]}, label: {LABELS[label_idx]}, reference length: {len(ref)}, prediction length: {len(pred)}'

    ################################################################################
    # Consider whole dataset
    ################################################################################
    label_to_ref_bio_tags_list = defaultdict(list)
    label_to_pred_bio_tags_list = defaultdict(list)
    for ref_bio_tags_list, pred_bio_tags_list in zip(all_ref_bio_tags_list, all_pred_bio_tags_list):
        if len(ref_bio_tags_list) != len(LABELS):
            print('ERROR: ref bio tags list')
        if len(pred_bio_tags_list) != len(LABELS):
            print('ERROR: pred bio tags list')

        for label, ref_bio_tags, pred_bio_tags in zip(LABELS, ref_bio_tags_list, pred_bio_tags_list):
            # x_i = [i for i in range(len(ref_bio_tags) - 1) if ref_bio_tags[i] == 'O' and 'I-' in ref_bio_tags[i + 1]]
            # y_i = [i for i in range(len(pred_bio_tags) - 1) if pred_bio_tags[i] == 'O' and 'I-' in pred_bio_tags[i + 1]]
            # if len(x_i) > 0 or len(y_i) > 0:
            #     import ipdb; ipdb.set_trace()
            label_to_ref_bio_tags_list[label].append(ref_bio_tags)
            label_to_pred_bio_tags_list[label].append(pred_bio_tags)

            if len(label_to_ref_bio_tags_list[label]) != len(label_to_pred_bio_tags_list[label]):
                print('ERROR: label_to_ref_pred_bio_tags')

    # Partial matching
    label_to_metrics = {}
    for label in label_to_ref_bio_tags_list.keys():
        ref_bio_tags_list = label_to_ref_bio_tags_list[label]
        pred_bio_tags_list = label_to_pred_bio_tags_list[label]
        if mode == 'exact':
            metrics = compute_metrics_exact(ref_bio_tags_list, pred_bio_tags_list, label)
        elif mode == 'partial':
            metrics = compute_metrics_partial(ref_bio_tags_list, pred_bio_tags_list, label)
        label_to_metrics[label] = metrics

    overall = calculate_macro_micro_weighted_metrics(label_to_metrics)
    # print(json.dumps(result, indent=2))
    with open(pred_dir / f'00_score_all_{mode}.json', 'w') as fd:
        formatted_overall = {}
        for k, v in overall.items():
            formatted_overall[k] = f'{v * 100:.2f}%'
        json.dump(formatted_overall, fd, indent=2)

        print(json.dumps(formatted_overall, indent=2))

    with open(pred_dir / f'00_score_{mode}.json', 'w') as fd:
        formatted_label_to_metrics = {}
        for label, metrics in label_to_metrics.items():
            metrics['precision'] = f"{metrics['precision'] * 100:.2f}%"
            metrics['recall'] = f"{metrics['recall'] * 100:.2f}%"
            metrics['f1'] = f"{metrics['f1'] * 100:.2f}%"
            formatted_label_to_metrics[label] = metrics
        json.dump(formatted_label_to_metrics, fd, indent=2)

        print(json.dumps(formatted_label_to_metrics, indent=2))
