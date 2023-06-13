import os
import pandas as pd
from tqdm import tqdm
from vietac import VACApiChecker
from utils import find_all_file_in_folder

checker = VACApiChecker(
    url="http://103.119.132.170:5052/api/v1/spell_checking"
)


def predict_on_file(file_path: str):
    output_path = file_path.replace("data", "prediction")
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        output_path = output_path.replace(".xlsx", ".csv")
    else:
        df = pd.read_csv(file_path)

    corrects = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_dict = row.to_dict()
        query = row_dict.get("clean_text").lower()
        output = checker.process(query).lower()
        corrects.append(output)

    if len(corrects) == len(df):
        df['VAC_corrected'] = corrects

    dir_path = os.path.dirname(os.path.realpath(output_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    folder_dir = "data/clean_text"
    file_paths = find_all_file_in_folder(folder_dir)
    for file in file_paths:
        print("Run on file", file)
        predict_on_file(file)
