__author__ = 'Titi Wei'
import json

# open txt file
def open_txt(path):
    with open(path, 'r') as f:
        lst = [line.strip() for line in f.readlines()]
    return lst

# open json file
import json
def open_json(path):
    with open(path, 'r') as f:
        return json.load(f)