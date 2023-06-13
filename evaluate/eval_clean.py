import pandas as pd
from tqdm import tqdm
from utils import find_all_file_in_folder
from unicodedata import normalize as nl
from metric import process


def check_correct(clean_text: str, vac_text: str):
    clean_text = process(clean_text)
    vac_text = process(vac_text)
    check = clean_text == vac_text
    return check


def eval_on_file(file_path):
    df = pd.read_csv(file_path)

    checks = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        clean_text = row["clean_text"]
        vac_text = row["VAC_corrected"]
        checks.append(check_correct(clean_text, vac_text))

    df['correct'] = checks
    wrong_c = len([x for x in checks if x is False])
    wrong_r = wrong_c / len(df)
    print(f'{file_path}| wrong count: {wrong_c}, wrong rate: {wrong_r}')
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    folder_dir = "prediction/clean_text"
    file_paths = find_all_file_in_folder(folder_dir)
    for file in file_paths:
        eval_on_file(file)
