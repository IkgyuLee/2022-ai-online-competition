import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
import json
from tqdm import tqdm
import shutil


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def convert_coco_json_to_yolo_txt(output_path, json_file):

    path = make_folders(output_path)

    with open(json_file, encoding='utf-8') as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "data.names")
    with open(label_file, "w", encoding='utf-8') as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w", encoding='utf-8') as f:
            for anno in anno_in_image:
                category = anno["category_id"] - 1
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")


def check_class_balance(class_target_list):
    cate = [0 for _ in range(14)]

    for train_labels in tqdm(class_target_list): # 1 txt file
        with open(train_labels, 'r', encoding='utf-8') as f:
            label = f.readlines() # 1 txt file labels
            for lab in label: # 1 line, in 1 txt file
                lab = lab.split(' ')
                cls = int(lab[0])

                if cls == 0:
                    cate[0] += 1
                elif cls == 1:
                    cate[1] += 1
                elif cls == 2:
                    cate[2] += 1
                elif cls == 3:
                    cate[3] += 1
                elif cls == 4:
                    cate[4] += 1
                elif cls == 5:
                    cate[5] += 1
                elif cls == 6:
                    cate[6] += 1
                elif cls == 7:
                    cate[7] += 1
                elif cls == 8:
                    cate[8] += 1
                elif cls == 9:
                    cate[9] += 1
                elif cls == 10:
                    cate[10] += 1
                elif cls == 11:
                    cate[11] += 1
                elif cls == 12:
                    cate[12] += 1
                elif cls == 13:
                    cate[13] += 1 

    result_cate = cate
    print(result_cate)

    X = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    Y = np.array(result_cate)

    # ax = sns.barplot(X,Y, order=X)
    # for p, q in zip(ax.patches, Y):
    #     ax.text(p.get_x()+p.get_width()/2.,
    #         p.get_height()*(1.01),
    #         "{}".format(q),
    #         ha = 'center'  )

    # plt.show()
    
    
def yolobbox2bbox(x, y, w, h, w_t, h_t):
    x1, y1 = (x - (w / 2.0)) * w_t, (y - (h / 2.0)) * h_t
    x2, y2 = (x + (w / 2.0)) * w_t, (y + (h / 2.0)) * h_t
    x1 = round(x1, 1)
    y1 = round(y1, 1)
    x2 = round(x2, 1)
    y2 = round(y2, 1)
    
    return (x1, y1, x2, y2)


def bbox2yolobbox(x1, y1, x2, y2, w_t, h_t):
    x = round((x2 + x1) / (2 * w_t), 6)
    y = round((y2 + y1) / (2 * h_t), 6)
    w = round((x2 - x1) / w_t, 6)
    h = round((y2 - y1) / h_t, 6)
    
    return (x, y, w, h)


def find_class_images(target_class, all_train_label_list):
    target_class = target_class
    num_class_target = 0
    class_target_list = []

    for train_labels in tqdm(all_train_label_list): # 1 txt file
        with open(train_labels, 'r', encoding='utf-8') as f:
            label = f.readlines() # 1 txt file labels
            for lab in label: # 1 line, in 1 txt file
                lab = lab.split(' ')
                cls, x, y, w, h = int(lab[0]), float(lab[1]), float(lab[2]), float(lab[3]), float(lab[4])

                if cls == target_class:
                    num_class_target += 1

                    class_target_list.append(train_labels)

    print("Class Num:", num_class_target)
    print("Image Num:", len(set(class_target_list)))
    
    return class_target_list


# Img Aug Function
def aug_images(img_path, txt_path, output_img_path, output_txt_path):
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    input_img = img[np.newaxis, :, :, :]

    txts = np.loadtxt(txt_path, dtype = str, delimiter = ' ').astype(float)
    if txts.ndim == 1:
        txts = txts[np.newaxis, :]
    
    labels = []
    for i, txt in enumerate(txts):
        x1, y1, x2, y2 = yolobbox2bbox(txt[1], txt[2], txt[3], txt[4], 1920, 1080)
        input_label = [int(txt[0]), x1, y1, x2, y2]
        labels.append(input_label)
    
    bbox = []
    for label in labels:
        bbox.append(ia.BoundingBox(x1 = label[1], y1 = label[2], x2 = label[3], y2 = label[4], label = label[0]))

    seq = iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.5, 0.7), "y": (0.5, 0.7)},
            rotate = (-15, 15)
        ),
        iaa.AdditiveGaussianNoise(scale = (0.05*255, 0.10*255)),
        iaa.GaussianBlur((0, 1.0)),
        iaa.PerspectiveTransform(scale=(0.01, 0.02))
    ])

    output_img, output_bbox = seq(images = input_img, bounding_boxes = bbox)
    output_img = np.squeeze(output_img, axis=0)

    result, encoded_img = cv2.imencode('.png', output_img)
    
    if result:
        with open(output_img_path, mode='w+b') as f:
            encoded_img.tofile(f)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for bbox in output_bbox:
            x, y, w, h = bbox2yolobbox(bbox.x1, bbox.y1, bbox.x2, bbox.y2, 1920, 1080)

            line = str(bbox.label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
            f.write(line)
            

if __name__ == "__main__":
    print("Folder Copy Started!")
    shutil.copytree("../../DATA/train", "../FINAL_DATA/train") # Directory Copy
    print("Folder Copy Finished!")
    
    if not os.path.isdir('../FINAL_DATA/train/labels'):
        os.mkdir('../FINAL_DATA/train/labels')
    
    convert_coco_json_to_yolo_txt("../FINAL_DATA/train/labels", "../FINAL_DATA/train/label/Train.json") # Convert Json to Txt
    
    print("Augmentation Started!")
    
    train_label_list = glob("../FINAL_DATA/train/labels/*.txt")

    class12_target_list = find_class_images(12, train_label_list)
    class10_target_list = find_class_images(10, train_label_list)
    class08_target_list = find_class_images(8, train_label_list)
    class03_target_list = find_class_images(3, train_label_list)
    class06_target_list = find_class_images(6, train_label_list)
    
    # Class 12(x 50), 10(x 10), 8(x 10), 3(x 5), 6(x 2), 2(x 2)
    class12_target_set = set(class12_target_list)
    class10_target_set = set(class10_target_list)
    class08_target_set = set(class08_target_list)
    class03_target_set = set(class03_target_list)
    class06_target_set = set(class06_target_list)
    
    # Seed Setting
    ia.seed(42)
    
    for i in tqdm(range(50)):
        for path in class12_target_set:
            txt_path = path
            img_path = path.replace("labels", "images").replace("txt", "png")

            aug_images(img_path, txt_path, img_path[:-4] + "_aug" + str(i) + ".png", txt_path[:-4] + "_aug" + str(i) + ".txt")

    for i in tqdm(range(10)):
        for path in class10_target_set:
            txt_path = path
            img_path = path.replace("labels", "images").replace("txt", "png")

            aug_images(img_path, txt_path, img_path[:-4] + "_aug" + str(i) + ".png", txt_path[:-4] + "_aug" + str(i) + ".txt")

    for i in tqdm(range(10)):
        for path in class08_target_set:
            txt_path = path
            img_path = path.replace("labels", "images").replace("txt", "png")

            aug_images(img_path, txt_path, img_path[:-4] + "_aug" + str(i) + ".png", txt_path[:-4] + "_aug" + str(i) + ".txt")

    for i in tqdm(range(5)):
        for path in class03_target_set:
            txt_path = path
            img_path = path.replace("labels", "images").replace("txt", "png")

            aug_images(img_path, txt_path, img_path[:-4] + "_aug" + str(i) + ".png", txt_path[:-4] + "_aug" + str(i) + ".txt")

    for i in tqdm(range(2)):
        for path in class06_target_set:
            txt_path = path
            img_path = path.replace("labels", "images").replace("txt", "png")

            aug_images(img_path, txt_path, img_path[:-4] + "_aug" + str(i) + ".png", txt_path[:-4] + "_aug" + str(i) + ".txt")
            
    all_train_label_list = glob("../FINAL_DATA/train/labels/*.txt")
    check_class_balance(all_train_label_list)
    print("Total Image Number: ", len(all_train_label_list)) 
    print("Augmentation Finished!")