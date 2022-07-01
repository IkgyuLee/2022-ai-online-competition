import os


if __name__ == "__main__":
    print("Model-1 Training Started!")
    os.system("python train.py --batch 8 --img 832 --epochs 16 --data '../FINAL_DATA/train/data.yaml' --name yolov5l6-model --weights yolov5l6.pt --save-period 1")
    print("Model-1 Training Finished!")
    
    print("Model-2 Training Started!")
    os.system("python train.py --batch 8 --img 704 --epochs 20 --data '../FINAL_DATA/train/data.yaml' --name yolov5x6-model --weights yolov5x6.pt --save-period 1")
    print("Model-2 Training Finished!")