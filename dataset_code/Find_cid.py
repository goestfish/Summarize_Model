import os
import re

folder_path = r'txtfiles'

def is_cid_format(text):
    return re.match(r'(\(cid:\d+\))+', text) is not None

with open('Cidfile.txt', 'w') as result_file:
    for root, ds, fs in os.walk(folder_path):
        for file_name in fs:
            if file_name.endswith('.txt'):
                with open(os.path.join(root, file_name), 'r', encoding='utf-8') as file:
                    content = file.read(30)
                    if is_cid_format(content):
                        result_file.write(os.path.join(root, file_name) + '\n')