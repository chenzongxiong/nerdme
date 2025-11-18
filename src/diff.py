from webanno_tsv import webanno_tsv_read_file
from collections import defaultdict
import pandas as pd
import os
import re
import cleaner
from predictor import BasePredictor, LABELS
from difflib import SequenceMatcher
from tabulate import tabulate
import logging


def get_min_starts_from_objects(tokens):
    min_starts = defaultdict(lambda: float('inf'))
    for token in tokens:
        sentence_idx = token.sentence_idx
        start = token.start
        if start < min_starts[sentence_idx]:
            min_starts[sentence_idx] = start
    return dict(sorted(min_starts.items()))


def insert_annotations_into_sentence(sentences, annotations, min_starts):
    results = []
    for sentence in sentences:
        text = sentence.text
        sentence_idx = sentence.idx
        min_start = min_starts.get(sentence_idx, 0)

        sentence_annotations = [
            anno for anno in annotations
            if any(token.sentence_idx == sentence_idx for token in anno.tokens)
        ]

        sentence_annotations.sort(key=lambda x: x.tokens[0].start)

        parts = []
        last_end = 0

        for annotation in sentence_annotations:
            start_idx = annotation.tokens[0].start - min_start
            end_idx = annotation.tokens[-1].end - min_start

            if start_idx > last_end:
                parts.append(text[last_end:start_idx])

            parts.append(f"<{annotation.label}>{text[start_idx:end_idx]}</{annotation.label}>")
            last_end = end_idx

        if last_end < len(text):
            parts.append(text[last_end:])

        annotated_text = ''.join(parts)
        results.append((sentence_idx, text, annotated_text))

    return results


def clean_annotation_tags(text, labels):
    """Remove all annotation tags from text using the provided labels"""
    for label in labels:
        text = re.sub(f'<{label}>', '', text, flags=re.IGNORECASE)
        text = re.sub(f'</{label}>', '', text, flags=re.IGNORECASE)
    return text


def process_original_file(base_path, file_name):
    """Load and process the TSV file to create annotated DataFrame"""
    file_path = os.path.join(base_path, file_name)
    doc = webanno_tsv_read_file(file_path)

    min_starts = get_min_starts_from_objects(doc.tokens)
    annotated_data = insert_annotations_into_sentence(doc.sentences, doc.annotations, min_starts)

    return pd.DataFrame(
        annotated_data,
        columns=['sentence_idx', 'original_sentence', 'annotated_sentence']
    )


def process_generated_files(df_original, model_name, file_name):
    """
    de_annotated_content - cleaned text without labels (for matching with original_sentence)
    cleaned_content - generated text with labels (for comparing with annotated_sentence)
    """
    generated_folder_path = f"../results/{model_name}/prompt-0/zzz_{file_name}"

    if not os.path.exists(generated_folder_path):
        print(f"Path does not exist: {generated_folder_path}")
        return pd.DataFrame(columns=['de_annotated_content', 'cleaned_content', 'source_file'])

    txt_files = [f for f in os.listdir(generated_folder_path) if f.endswith('.txt')]
    if not txt_files:
        print("No .txt files found in the folder")
        return pd.DataFrame(columns=['de_annotated_content', 'cleaned_content', 'source_file'])

    processed_data = []
    for txt_file in txt_files:
        try:
            with open(os.path.join(generated_folder_path, txt_file), 'r', encoding='utf-8') as f:
                content = f.read()

            cleaned = cleaner.Cleaner(content).clean()
            de_annotated = clean_annotation_tags(cleaned, LABELS)

            processed_data.append({
                'de_annotated_content': de_annotated,
                'cleaned_content': cleaned,
                'source_file': os.path.join(generated_folder_path, txt_file)
            })

        except Exception as e:
            print(f"Error reading file {txt_file}: {str(e)}")

    if not processed_data:
        print("No files were successfully read")
        return pd.DataFrame(columns=['de_annotated_content', 'cleaned_content', 'source_file'])

    # Create DataFrame with the three required columns
    df_generated = pd.DataFrame(processed_data)

    return df_generated


def find_best_matches(df_original, df_generated):
    """Find the best matching generated content for each original sentence"""
    matched_data = []

    for idx, row in df_original.iterrows():
        original_text = row['original_sentence']
        best_match = None
        best_score = 0  # Ignore matches with similarity

        for gen_idx, gen_row in df_generated.iterrows():
            similarity = SequenceMatcher(None, original_text, gen_row['de_annotated_content']).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match = gen_row

        if best_match is not None:
            matched_data.append({
                'sentence_idx': row['sentence_idx'],
                'original_sentence': original_text,
                'original_annotated': row['annotated_sentence'],
                'de_annotated_content':best_match['de_annotated_content'],
                'generated_annotated': best_match['cleaned_content'],
                'source_file': best_match['source_file'],
                'similarity_score': best_score   # the similarity score between "original sentence" and "de_annotated_content"
            })

    return pd.DataFrame(matched_data)


# def compare_annotations(df_matched):
#     """Compare original human-annotated sentence and deepseek-generated annotated sentence and return only rows with differences"""
#     diff_rows = []
#
#     for idx, row in df_matched.iterrows():
#         original = row['original_annotated']
#         generated = row['generated_annotated']
#
#         # Compare the text with annotations removed
#         original_clean = clean_annotation_tags(original, LABELS)
#         generated_clean = clean_annotation_tags(generated, LABELS)
#
#         if original_clean != generated_clean:
#             diff_rows.append(row)
#         else:
#             # Also check if the annotations themselves are different
#             original_anno_only = original.replace(original_clean, '')
#             generated_anno_only = generated.replace(generated_clean, '')
#             if original_anno_only != generated_anno_only:
#                 diff_rows.append(row)
#
#     return pd.DataFrame(diff_rows)


def main():
    # Configuration
    phase = 'train'
    base_path = f'../data/{phase}'
    model_name = 'deepseek-chat'

    # Initialize logging
    log_path = '../diff/comparison.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    # Get all TSV files in the directory
    tsv_files = [fp for fp in os.listdir(base_path)
                 if os.path.isfile(os.path.join(base_path, fp)) and fp.endswith('.tsv')]

    if not tsv_files:
        print(f"No TSV files found in {base_path}")
        return

    # Create diff directory if it doesn't exist
    os.makedirs('../diff', exist_ok=True)

    # Initialize total counters
    total_sentences = 0
    total_diff = 0

    # Process each TSV file
    for file_name in tsv_files:
        logging.info(f"\nProcessing file: {file_name}")

        try:
            df_original = process_original_file(base_path, file_name)
            df_generated = process_generated_files(df_original, model_name, file_name)

            if df_generated.empty:
                logging.warning(f"No generated content found for {file_name}")
                continue

            # Find best matches between original and generated content
            df_matched = find_best_matches(df_original, df_generated)

            if df_matched.empty:
                logging.warning(f"No matches found for {file_name}")
                continue

            # Compare annotations and get only rows with differences
            # df_diff = compare_annotations(df_matched)
            df_diff = df_matched[df_matched['similarity_score'] != 1.0]

            # Update total counters
            matched_count = len(df_matched)
            diff_count = len(df_diff)
            total_sentences += matched_count
            total_diff += diff_count

            # Calculate difference percentage for this file
            diff_percentage = (diff_count / matched_count * 100) if matched_count > 0 else 0

            # Log file statistics
            logging.info(f"\nStatistics for {file_name}:")
            logging.info(f"- Total sentences: {matched_count}")
            logging.info(f"- Changed sentences: {diff_count}")
            logging.info(f"- Difference percentage: {diff_percentage:.2f}%")

            # Save results with base filename (without .tsv extension)
            base_name = os.path.splitext(file_name)[0]
            df_matched.to_csv(f'../diff/all_{base_name}.csv', index=False)
            df_diff.to_csv(f'../diff/diff_{base_name}.csv', index=False)

            # Print differences in tabular format
            if not df_diff.empty:
                print(f"\nDifferences found in {file_name}:")
                print(tabulate(
                    df_diff[['original_annotated', 'generated_annotated']],
                    headers=['Original Annotations', 'Generated Annotations'],
                    tablefmt='grid',
                    showindex=False
                ))
            else:
                print(f"No differences found in {file_name}")

        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
            continue

    # Final summary
    logging.info("\n\n=== FINAL SUMMARY ===")
    logging.info(f"Total processed sentences: {total_sentences}")
    logging.info(f"Total changed sentences: {total_diff}")

    if total_sentences > 0:
        overall_diff_percent = (total_diff / total_sentences) * 100
        logging.info(f"Overall difference percentage: {overall_diff_percent:.2f}%")
    else:
        logging.info("No sentences processed - cannot calculate overall percentage")

if __name__ == "__main__":
    main()
