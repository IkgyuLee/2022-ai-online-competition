import os
import json
import shutil
from tqdm import tqdm


def make_submission_file(prediction_path):
    with open('../../DATA/Test_Images_Information.json', 'r', encoding='utf-8') as f:
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
    
    if not os.path.isdir('../submissions'):
        os.mkdir('../submissions')
    
    with open('../submissions/reproduction.json', 'w', encoding='utf-8') as f:
        json.dump(submission_jdict, f)

if __name__ == "__main__":
    
    if not os.path.isdir('../submissions'):
        os.mkdir('../submissions')
        
    # weights = ['runs/train/yolov5l6-model/weights/epoch14.pt', 'runs/train/yolov5l6-model/weights/epoch15.pt', 'runs/train/yolov5x6-model/weights/epoch19.pt']
    # weights = ['../submissions/leaderboard_weight1.pt', '../submissions/leaderboard_weight2.pt', '../submissions/leaderboard_weight3.pt']
    # for weight in weights:
    #     shutil.copy(weight, '../submissions')
    #
    os.system("python val.py --data '../FINAL_DATA/train/data.yaml' --img 1280 --batch-size 16 --conf-thres 0.050 --iou-thres 0.65 --weights '../submissions/leaderboard_weight1.pt' '../submissions/leaderboard_weight2.pt' '../submissions/leaderboard_weight3.pt' --name yolov5l6x6-test --task test --save-json --save-conf")
    
    make_submission_file("./runs/val/yolov5l6x6-test/epoch14_predictions.json")
