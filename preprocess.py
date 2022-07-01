import torch
import numpy as np
import random
import os

from glob import glob
from sklearn.model_selection import train_test_split
import yaml
import ast
from IPython.core.magic import register_line_cell_magic


def my_seed_everywhere(seed: int = 42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


if __name__ == "__main__":
    print("Preprocessing Started!")
    
    my_seed_everywhere(42)

    # Prepare Datasets
    img_list = glob('../FINAL_DATA/train/images/*.png')
    print("Total Image Number:",len(img_list))
    train_img_list, val_img_list = train_test_split(img_list, test_size=0.05, random_state=42)
    print("Train and Val Image Number:", len(train_img_list), len(val_img_list))

    with open('../FINAL_DATA/train/train.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(train_img_list) + '\n')

    with open('../FINAL_DATA/train/val.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(val_img_list) + '\n')

    data = {}
    data['train'] = '../FINAL_DATA/train/train.txt'
    data['val'] = '../FINAL_DATA/train/val.txt'
    data['test'] = '../../DATA/test/images'
    data['nc'] = 14
    data['names'] = ['세단(승용차)', 'SUV', '승합차', '버스', '학원차량(통학버스)', '트럭', '택시', '성인', '어린이', '오토바이', '전동킥보드', '자전거', '유모차', '쇼핑카트']


    with open('../FINAL_DATA/train/data.yaml', 'w', encoding='utf8') as f:
        yaml.dump(data, f)

    print(data)

    with open("../FINAL_DATA/train/data.yaml", 'r') as stream:
        names = str(yaml.safe_load(stream)['names'])

    namesFile = open("../FINAL_DATA/train/data.names", "w+", encoding='utf8')
    names = ast.literal_eval(names)
    for name in names:
        namesFile.write(name +'\n')
    namesFile.close()

    print("Preprocessing Finished!")