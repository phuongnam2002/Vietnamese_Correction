import random
import string
import pandas as pd
import tqdm
from tqdm import tqdm

def repeat_last_char(word):
    return word + word[-1]

# def missing_last_char(word):
#     if len(word) > 3:
#         return word[:-1]
#     else:
#         return word
    
def telex_error(word):
    telex_dict = {"ă": "aw", "â": "aa", "ê": "ee", "ễ": "eex", "ô": "oo", "ậ": "aaj", "ộ": "ooj","ề": "eef",
                  "ắ": "aws", "ế": "ees", "ư": "u", "ơ": "ow", "ứ": "uws", "ờ": "owf", "á": "as", "ì": "if",
                  "ó": "os", "ỏ": "or", "ạ": "aj", "ệ": "eej", "ú": "us", "ả": "ar", "ã": "ax", "à": "af", "ự": "uwj"}
    new_word = ""
    for i in range(len(word)):
        if word[i] in telex_dict:
            new_word += telex_dict[word[i]]
        else:
            new_word += word[i]
    return new_word

def add_accent(word):
    accent_dict = {"a": "àáảãạ", "ă": "ằắẳẵặ", "â": "ầấẩẫậ", "e": "èéẻẽẹ", "ê": "ềếểễệ", "i": "ìíỉĩị", "o": "òóỏõọ", "ô": "ồốổỗộ", "ơ": "ờớởỡợ", "u": "ùúủũụ", "ư": "ừứửữự", "y": "ỳýỷỹỵ"}
    vowels = "aeiouy"
    new_word = ""
    for i in range(len(word)):
        if word[i] in vowels:
            if i > 0 and word[i-1] in accent_dict:
                new_word += accent_dict[word[i-1]][-1]
            else:
                new_word += accent_dict[word[i]][0]
        else:
            new_word += word[i]
    return new_word

def miss_character(word):
    character_list = {"ức": "ứ", "ua": "a", "iế": "í", "ọa": "ọ", "ườ": "ừ", "ng": "n", "iễ": "ỉn", "ấu": "áu", "ứn": "ún", "ây": "ay", "ần": "àn", 
                      "iệ": "ị", "ăn": "an", "ào": "à", "ạo": "ạ", "ạn": "an", "ất": "át", "iề": "ìu", "ọc": "oc", "ồm": "òm", "ầu": "àu", "ôn": "on",
                      "ươ": "uo", "ín": "in", "oạ": "ọ", "iê": "i", "iễ": "ĩ", "yê": "ye", "ái": "ai"}
    new_word = ""
    for i in range(len(word)):
        if word[i:i+2] in character_list:
            new_word += character_list[word[i:i+2]]
        else:
            new_word += word[i]
    return new_word

def diacritics_error(word):
    diacritic_dict = {"o": "ô", "a": "â", "ụ": "ự", "ế": "é", "á": "a", "ẻ": "é", "â": "a",
                      "ố": "ó", "ó": "o", "ấ": "á", "ộ": "ọ", "ặ": "ạ", "í": "i"}
    new_word = ""
    for i in range(len(word)):
        if word[i] in diacritic_dict:
            new_word += diacritic_dict[word[i]]
        else:
            new_word += word[i]
    return new_word

list_docs = ["sinh_hoc.csv"]

for name in list_docs:
    print(f"doing in file : {name}")
    make_error = []
    docs = pd.read_csv(name)
    text = docs["clean_text"]
    for i in tqdm(range(len(text))):
        new_sentence = []
        sentence = text[i]

        token_list = sentence.split()
        length = len(sentence.split())

        error_number = random.sample(range(1, length), random.randint(2, 3))
        for j in error_number:
            error_func = random.choice([repeat_last_char, miss_character, 
                                        telex_error, add_accent, diacritics_error])

            token_list[j] = error_func(token_list[j])
        token_list[len(error_number)-1] = telex_error(token_list[len(error_number)-1])
        new_sentence = " ".join(token_list)
        make_error.append(new_sentence)
    docs["error_text"] = make_error
    for i in range(len(text)):
        if docs['clean_text'][i] == docs['error_text'][i]:
            docs = docs.drop(labels = i, axis= 0)

    print(len(docs))
    docs.to_csv(f"error_data/{name}")
    print("------------------------DONE-------------------------")

