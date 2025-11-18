from functools import reduce
from collections import defaultdict
from webanno_tsv import Document, Annotation, Token, Sentence
from typing import List
from dataclasses import replace
from copy import deepcopy

LABEL_IDS = defaultdict(int)
TOKEN_LABEL_IDS = {}


def get_label_id(label: str) -> int:
    global LABEL_IDS
    lid = LABEL_IDS[label]
    LABEL_IDS[label] += 1
    return lid


def get_layer_name():
    return 'de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity'


def get_field_name():
    return 'value'


def make_annotation(tokens: List[Token], label: str) -> Annotation:
    global TOKEN_LABEL_IDS

    key = f'{tokens[0].start},{tokens[-1].end},{label}'
    if key not in TOKEN_LABEL_IDS:
        lid = get_label_id(label)
        TOKEN_LABEL_IDS[key] = lid
    else:
        lid = TOKEN_LABEL_IDS[key]

    return Annotation(
        tokens=tokens,
        label=label,
        layer=get_layer_name(),
        field=get_field_name(),
        label_id=lid
    )


def make_webanno_document(sentences: List[Sentence], tokens: List[Token], annotations: List[Annotation]) -> Document:
    layer_defs = [('de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity', ['identifier', 'value'])]

    annotations = reduce(lambda acc, item: acc if any(item.should_merge(existing) for existing in acc) else acc + [item], annotations, [])
    return Document(
        layer_defs=layer_defs,
        sentences=sentences,
        tokens=tokens,
        annotations=annotations,
    )


def replace_webanno_annotations(doc: Document, annotations: List[Annotation]) -> Document:
    doc = replace(doc, annotations=annotations)
    return doc


def make_span_tokens(tokens: List[Token], start_char: int, end_char: int) -> List[Token] | None:
    '''
    Example: Given a list of token List[Token] = [(Few-Shot_ED.json.zip, 86, 107)],
    the correct annotation is `<DATASET>Few-Shot_ED</DATASET>.json.zip`, the start char is 86, end char is 97.
    We need to generate the correct span tokens e.g. [(Few-Shot_ED, 86, 97)].
    Using the function can achieve this functionality
    '''
    base_start = tokens[0].start
    start_char, end_char = start_char + base_start, end_char + base_start
    span_tokens = []
    added_tid = set()
    for token in tokens:
        if token.end < start_char:
            continue
        if token.start > end_char:
            continue
        if token.idx in added_tid:
            continue

        span_tokens.append(token)
        added_tid.add(token.idx)

    if len(span_tokens) == 0:
        return None

    span_tokens_for_debug = deepcopy(span_tokens)
    # print('span tokens for debug: ', span_tokens_for_debug)
    # print('----------------------------------------')
    first, last = span_tokens[0], span_tokens[-1]
    # NOTE: only the first and last tokens are in corner case
    span_tokens = span_tokens[1:-1]
    if first.idx == last.idx:           # only one tokens
        token = Token(idx=f'{last.idx}.99', sentence_idx=last.sentence_idx, start=start_char, end=end_char, text=first.text[start_char-first.start:end_char-first.start])
        span_tokens.append(token)
    else:
        if first.start <= start_char:
            if first.start == start_char:
                span_tokens.insert(0, first)
            else:
                text = first.text[start_char - first.start:]
                if len(text) > 0:
                    token = Token(idx=f'{first.idx}.99', sentence_idx=first.sentence_idx, start=start_char, end=first.end, text=text)
                    span_tokens.insert(0, token)

        if last.end >= end_char:
            if last.end == end_char:
                span_tokens.append(last)
            else:
                text = last.text[:end_char-last.start]
                if len(text) > 0:
                    token = Token(idx=f'{last.idx}.99', sentence_idx=last.sentence_idx, start=last.start, end=end_char, text=text)
                    span_tokens.append(token)


    # if len(span_tokens) == 0:
    #     return None

    return span_tokens, span_tokens_for_debug
