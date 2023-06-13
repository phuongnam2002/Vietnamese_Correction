import pandas as pd
import numpy as np
from tqdm import tqdm

from metric import process, calculate_precision, calculate_recall
from utils import find_all_file_in_folder


def eval_on_file(file_path):
    df = pd.read_csv(file_path)
    precision = []
    recall = []

    for j in tqdm(range(len(df['clean_text']))):
        if process(df['clean_text'][j]).split() == process(df['error_text'][j]).split():
            df = df.drop(labels=j, axis=0)
        else:
            correct_sentence = df['clean_text'][j]
            incorrect_sentence = df['error_text'][j]
            corrected_sentence = df['VAC_corrected'][j]

            output1 = calculate_precision(correct_sentence, incorrect_sentence, corrected_sentence)
            output2 = calculate_recall(correct_sentence, incorrect_sentence, corrected_sentence)
            precision.append(output1)
            recall.append(output2)

    df['precision'] = precision
    df['recall'] = recall
    a_precision = sum(precision)/len(precision)
    a_recall = sum(recall) / len(recall)
    print(f"{file_path}| precision: {a_precision}, recall: {a_recall}")
    df.to_csv(file_path)
    print('------------------------------')


if __name__ == '__main__':
    folder_dir = "prediction/error_text"
    file_paths = find_all_file_in_folder(folder_dir)
    for file in file_paths:
        print("Eval on file", file)
        eval_on_file(file)
