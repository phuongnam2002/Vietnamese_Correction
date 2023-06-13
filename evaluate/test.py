import pandas as pd
import numpy as np
from utils import find_all_file_in_folder
from unicodedata import normalize as nl


def process(query: str):
    query = nl('NFKC', query)
    punctuation = [",", ".", "?", "!", ":", "/", "…", "`", ";", "_", '"', '"',
                   "(", ")", "-", "\\", '"', '”', '“', "–", "'", "."]
    query = query.lower()
    for i in punctuation:
        query = query.replace(i, ' ')
    query = " ".join(query.split())
    query = query.strip()
    return query


def check_correct(clean_text: str, vac_text: str):
    clean_text = process(clean_text).encode("utf-8")
    vac_text = process(vac_text).encode("utf-8")
    check = clean_text == vac_text
    return check


def eval_on_file(file):
    df = pd.read_excel(file)
    precision = df['precision'].values.copy()
    recall = df['recall'].values.copy()

    precision = np.mean(precision)
    recall = np.mean(recall)

    print(f"Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    folder_dir = 'prediction/error_text'
    files = find_all_file_in_folder(folder_dir)
    for file in files:
        print('Eval on file:', file)
        eval_on_file(file)
