import os
import json
from tqdm import tqdm


def make_submission_file(prediction_path, num):
    with open('../dataset/download1-20220607T050736Z-003/download1/test/Test_Images_Information.json', 'r', encoding='utf-8') as f:
        test_info = json.load(f)

    with open(prediction_path, 'r', encoding='utf-8') as f:
        prediction_json = json.load(f)

    submission_jdict = []

    for prediction in tqdm(prediction_json, desc='submission_json'):
        for test in test_info['images']:
            if test['file_name'].split('.')[0] == prediction['image_id']:
                submission_jdict.append({'image_id': test['id'],
                                        'category_id': prediction['category_id'] + 1,
                                        'bbox': prediction['bbox'],
                                        'score': prediction['score'],
                                        })

    with open(f'../submissions/submission{num}.json', 'w', encoding='utf-8') as f:
        json.dump(submission_jdict, f)

path1 = "./runs/val/yolov5l6x6-test1/epoch13_predictions.json"
path2 = "./runs/val/yolov5l6x6-test2/epoch13_predictions.json"
path3 = "./runs/val/yolov5l6x6-test3/epoch13_predictions.json"

os.system("python val.py --data '../dataset/train-001/data.yaml' --img 1280 --batch-size 16 --conf-thres 0.025 --iou-thres 0.65 --weights 'runs/train/yolov5l6-220615/weights/best.pt' --name yolov5l6x6-test1 --task test --save-json --save-conf")
make_submission_file(path1, 1)

os.system("python val.py --data '../dataset/train-001/data.yaml' --img 1280 --batch-size 16 --conf-thres 0.020 --iou-thres 0.65 --weights 'runs/train/yolov5l6-220615/weights/best.pt' --name yolov5l6x6-test2 --task test --save-json --save-conf")
make_submission_file(path1, 2)

os.system("python val.py --data '../dataset/train-001/data.yaml' --img 1280 --batch-size 16 --conf-thres 0.150 --iou-thres 0.65 --weights 'runs/train/yolov5l6-220615/weights/best.pt' --name yolov5l6x6-test3 --task test --save-json --save-conf")
make_submission_file(path1, 3)