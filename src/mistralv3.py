import re
import sys
import os
import random
from collections import defaultdict
from termcolor import colored
from openai import OpenAI
from functools import partial, reduce
import operator as op
import hashlib
import multiprocessing as mp
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import utils
import cleaner
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
from predictor import BasePredictor, LABELS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MistralV3Predictor(BasePredictor):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        self.prompt_id = 0
        prompt_template_path = f'config/{model_name}-prompt-{self.prompt_id}.txt'
        if os.path.isfile(prompt_template_path):
            with open(prompt_template_path, 'r') as fd:
                self.prompt_template = fd.read()
        else:
            raise
        self.parallel = False
        self.mismatch_sentences = 0
        self.total_sentences = 0

        self.tokenizer = None
        self.model = None

    def do_prediction(self, sentence, sid_path):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        print(f"Process-{os.getpid()} processing {colored(sentence.text, 'red')} ...")
        prompt = self.prompt_template.replace('{input_text}', sentence.text)

        messages = [
            {"role": "system", "content": "You are a helpful NER annotator."},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            # max_new_tokens=255,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)

        with open(sid_path, 'w') as file:
            file.write(result)

    def set_file_name(self, file_name):
        self.file_name = file_name


def double_check(ref_doc, predicted_doc, file_name):
    if ref_doc.text != predicted_doc.text:
        logging.warning(f'{file_name} content changed')
    if len(ref_doc.sentences) != len(predicted_doc.sentences):
        logging.warning(f'{file_name} sentences changed, {len(ref_doc.sentences)}/{len(predicted_doc.sentences)}')
    if len(ref_doc.tokens) != len(predicted_doc.tokens):
        logging.debug(f'{file_name} tokens changed')
    for s1, s2 in zip(ref_doc.sentences, predicted_doc.sentences):
        if s1 != s2:
            logging.warning(f'{file_name} sentence changed, \n{s1}\n{s2}')

    for t1, t2 in zip(ref_doc.tokens, predicted_doc.tokens):
        if t1 != t2:
            logging.warning(f'token changed: \n{t1}\n{t2}')


if __name__ == "__main__":
    phase = 'test_unlabeled'
    base_path = Path(f'data/{phase}')
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model_name = model_id.split('/')[-1]
    file_paths = sorted([x for x in base_path.rglob('*.tsv')])
    output_folder = Path(f'results/{model_name}/{phase}')
    os.makedirs(output_folder, exist_ok=True)

    predictor = MistralV3Predictor(
        model_id=model_id
    )

    for file_path in file_paths:
        print(f'file_name: {file_path.name}')

        # if 'prasunroy_air-writing_master_README.md.tsv' not in file_path.name:
        #     continue
        predictor.set_file_name(file_path.name)
        ref_doc = webanno_tsv_read_file(file_path)
        pred_doc = predictor(ref_doc)
        double_check(ref_doc, pred_doc, file_path.name)
        prediction_path = output_folder / file_path.name
        with open(prediction_path, 'w') as fd:
            fd.write(pred_doc.tsv())
