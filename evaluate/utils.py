import os
import json
import re
from typing import Text, Union, List, Dict
import yaml


def block_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='>')


def dict_presenter(dumper, data: Dict):
    return dumper.represent_dict(data.items())


class literal(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')


def fold_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='>')


def write_yaml(data, file_path,
               block_present=False,
               fold_present=False,
               dict_represent=False,
               **kwargs):
    encoding = kwargs.get("encoding", "utf-8")

    if block_present:
        yaml.add_representer(literal, literal_presenter)

    if fold_present:
        yaml.add_representer(literal, fold_presenter)

    if dict_represent:
        yaml.add_representer(dict, dict_presenter)

    with open(file_path, 'w', encoding=encoding) as pf:
        yaml.dump(data, pf, allow_unicode=True,
                  default_flow_style=False, sort_keys=False,
                  **kwargs
                  )


def load_yaml(file_path, **kwargs) -> Dict:
    encoding = kwargs.get("encoding", "utf-8")
    with open(file_path, 'r', encoding=encoding) as pf:
        return yaml.load(pf, Loader=yaml.SafeLoader)


def write_json(data, file_path, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    with open(file_path, 'w', encoding=encoding) as pf:
        json.dump(data, pf, ensure_ascii=False, indent=4)


def load_json(file_path, **kwargs) -> Dict:
    encoding = kwargs.get("encoding", "utf-8")

    with open(file_path, 'r', encoding=encoding) as pf:
        return json.load(pf)


def check_format_file(file: Text, format_file: Union[List[Text], Text]):
    if isinstance(format_file, Text):
        format_file = [format_file]

    for format_f in format_file:
        if file.endswith(format_f):
            return True

    return False


def find_all_file_in_folder(folder_path, output: list = None, format_file: Union[List[Text], Text] = None):
    file_names = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, f) for f in file_names]
    if output is None:
        output = []
    for f in file_paths:
        if os.path.isfile(f):
            output.append(f)
        elif os.path.isdir(f):
            find_all_file_in_folder(f, output)
    if format_file:
        output = [f for f in output if check_format_file(f, format_file)]
    return output


def open_file(file_path: str):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [example.rstrip() for example in data]


def write_file(data: list, file_path):
    with open(file_path, 'w') as file:
        for example in data:
            file.write(f"{example}\n")

