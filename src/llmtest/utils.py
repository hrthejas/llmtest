import os
from langchain.prompts import load_prompt


def read_prompt_text(prompt_file_path,default_template):
    if os.path.isfile(prompt_file_path):
        print("Reading prompt file " + prompt_file_path)
        text_file = open(prompt_file_path, "r")
        data = text_file.read()
        text_file.close()
        print(data)
        return data
    else:
        print("Cannot open prompt file " + prompt_file_path + " defaulting to default prompt")
        return default_template


def load_prompt(prompt_file_path):
    prompt = load_prompt(prompt_file_path)
    print(prompt.value)
    return prompt
