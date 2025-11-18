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

import utils
import cleaner
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
from base_predictor import GenerativePredictor, LABELS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DeepSeekChat(GenerativePredictor):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
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

    def do_prediction(self, sentence, sid_path):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            print(f"Process-{os.getpid()} processing {colored(sentence.text, 'red')} ...")
            prompt = self.prompt_template.replace('{input_text}', sentence.text)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=4096,
            )
            print(f"Process-{os.getpid()} predict {colored(sentence.text, 'cyan')} successfully")
            result = response.choices[0].message.content
            with open(sid_path, 'w') as file:
                file.write(result)
        except Exception as ex:
            logging.error(f'[do_prediction] got exception: {ex}')

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
    file_paths = sorted([x for x in base_path.rglob('*.tsv')])
    model_name = 'deepseek-chat'
    output_folder = Path(f'results/{model_name}/{phase}')
    os.makedirs(output_folder, exist_ok=True)
    # DeepSeek Chat
    predictor = DeepSeekChat(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url='https://api.deepseek.com',
        model_name=model_name
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
