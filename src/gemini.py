import re
import sys
import os
import random
from collections import defaultdict
from termcolor import colored
import logging
from pathlib import Path
import google.generativeai as genai

import utils
import cleaner
from webanno_tsv import webanno_tsv_read_file, Document, Annotation, Token
from base_predictor import GenerativePredictor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prompt_gemini(client, prompt, image_path=None, demonstrations=None):
    """
    Prompt Gemini with optional in-context learning using demonstrations.

    Args:
        client: The initialized Gemini model client
        prompt (str): The text prompt to send to Gemini
        image_path (str, optional): Path to an image file to include in the prompt
        demonstrations (list, optional): List of tuples (prompt, image_path, answer) for in-context learning

    Returns:
        str: Gemini's response
    """
    # import PIL.Image

    # Initialize the content list that will be sent to Gemini
    content_parts = []
    # # Add demonstrations for in-context learning if provided
    # if demonstrations and len(demonstrations) > 0:
    #     # Format the demonstrations
    #     for demo_prompt, demo_image_path, demo_answer in demonstrations:
    #         # Add a separator for each demonstration
    #         content_parts.append("Example Input:")

    #         # Add demonstration image if provided
    #         if demo_image_path:
    #             try:
    #                 demo_image = PIL.Image.open(demo_image_path)
    #                 content_parts.append(demo_image)
    #             except Exception as e:
    #                 content_parts.append(f"[Image could not be loaded: {str(e)}]")

    #         # Add demonstration prompt
    #         content_parts.append(demo_prompt)

    #         # Add demonstration answer
    #         content_parts.append("Example Output:")
    #         content_parts.append(demo_answer)

    #         # Add a separator between demonstrations
    #         content_parts.append("---")

    #     # Add a final separator before the actual query
    #     content_parts.append("Now, please analyze the following:")

    # # Add the current image if provided
    # if image_path:
    #     try:
    #         image = PIL.Image.open(image_path)
    #         content_parts.append(image)
    #     except Exception as e:
    #         return f"Error loading image: {str(e)}"

    # Add the current prompt
    content_parts.append(prompt)

    # Generate the response
    # try:
    #     response = client.model.generate_content(content_parts, generation_config={"temperature": 0.0})
    #     return response.text
    # except Exception as e:
    #     return f"Error generating response: {str(e)}"

    response = client.generate_content(content_parts, generation_config={"temperature": 0.0})
    return response.text


class GeminiFlash(GenerativePredictor):
    def __init__(self, model_name: str):
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
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(self.model_name)

    def do_prediction(self, sentence, sid_path):
        try:
            print(f"Process-{os.getpid()} processing {colored(sentence.text, 'red')} ...")
            prompt = self.prompt_template.replace('{input_text}', sentence.text)
            response = prompt_gemini(self.client, prompt)
            print(f"Process-{os.getpid()} predict {colored(sentence.text, 'cyan')} successfully")
            result = response
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
    model_name = 'gemini-2.0-flash'
    output_folder = Path(f'results/{model_name}/{phase}')
    os.makedirs(output_folder, exist_ok=True)

    predictor = GeminiFlash(model_name=model_name)

    for idx, file_path in enumerate(file_paths):
        print(f'file_name: {file_path.name}')

        predictor.set_file_name(file_path.name)
        ref_doc = webanno_tsv_read_file(file_path)
        pred_doc = predictor(ref_doc)
        double_check(ref_doc, pred_doc, file_path.name)
        prediction_path = output_folder / file_path.name
        with open(prediction_path, 'w') as fd:
            fd.write(pred_doc.tsv())
