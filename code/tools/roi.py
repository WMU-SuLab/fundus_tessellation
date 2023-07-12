import numpy as np
import cv2
from PIL import Image
import random
import json
import matplotlib.pyplot as plt

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "not find data for train"
    assert len(val_images_path) > 0, "not find data for eval"

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def get_grade1_cut(img, f_p):  # cut original image to grade1 Roi
    h, w = img.shape[:2]
    r0 = h / 25.4
    r = r0 * 6
    image_part = img.copy()
    cv2.circle(image_part, (int(f_p[0]), int(f_p[1])), int(r), (0, 0, 0), thickness=-1)
    image_part = image_part[:, int(w / 2) - int(h / 2):int(w / 2) + int(h / 2)]
    return image_part


def get_grade2_cut(img, f_p):  # cut original image to grade2 Roi
    h, w = img.shape[:2]
    r0 = h / 25.4
    r1 = r0 * 3
    r2 = r0 * 6
    image_part = img.copy()
    cv2.circle(image_part, (int(f_p[0]), int(f_p[1])), int(r1), (0, 0, 0), thickness=-1)
    center = (int(f_p[0]), int(f_p[1]))
    mask = create_circular_mask(h, w, center=center, radius=int(r2))
    masked_img = image_part.copy()
    masked_img[~mask] = 0
    masked_img = masked_img[center[1] - int(r2):center[1] + int(r2), center[0] - int(r2):center[0] + int(r2)]
    return masked_img


def get_grade3_cut(img, f_p):  # cut original image to grade3 Roi
    h, w = img.shape[:2]
    r0 = h / 25.4
    r1 = r0 * 3
    image_part = img.copy()
    cv2.circle(image_part, (int(f_p[0]), int(f_p[1])), int(r0), (0, 0, 0), thickness=-1)
    center = (int(f_p[0]), int(f_p[1]))
    mask = create_circular_mask(h, w, center=center, radius=r1)
    masked_img = image_part.copy()
    masked_img[~mask] = 0
    masked_img = masked_img[center[1] - int(r1):center[1] + int(r1), center[0] - int(r1):center[0] + int(r1)]
    return masked_img


def get_grade4_cut(img, f_p):  # cut original image to grade4 Roi
    h, w = img.shape[:2]
    r0 = h / 25.4
    image_part = img.copy()
    center = (int(f_p[0]), int(f_p[1]))
    mask = create_circular_mask(h, w, center=center, radius=r0)
    masked_img = image_part.copy()
    masked_img[~mask] = 0
    masked_img = masked_img[center[1] - int(r0):center[1] + int(r0), center[0] - int(r0):center[0] + int(r0)]
    return masked_img


def cut_roi(img_path, fovea, grade, save_path):
    ori = np.array(Image.open(img_path))
    if grade == 'grade1':
        img_in = get_grade1_cut(img=ori, f_p=fovea)
    elif grade == 'grade2':
        img_in = get_grade2_cut(img=ori, f_p=fovea)
    elif grade == 'grade3':
        img_in = get_grade3_cut(img=ori, f_p=fovea)
    elif grade == 'grade4':
        img_in = get_grade4_cut(img=ori, f_p=fovea)
    else:
        img_in = ori
    Image.fromarray(img_in).save(save_path)


def run_circle(img_dir, f_resh, str_dir):
    img = cv2.imread(img_dir)
    h, w = img.shape[:2]
    r0 = h / 25.4
    r1 = 3 * r0
    r2 = 6 * r0

    cv2.circle(img, (int(f_resh[0]), int(f_resh[1])), int(r0), (255, 255, 255),
               thickness=5)  # thivknessÊÇ¸ù¾ÝÖÐ¼äµÄÏßÌõÔÚÁ½±ß½øÐÐÑÓÉì×öÏßÌõ´ÖÏ¸µÄ
    cv2.circle(img, (int(f_resh[0]), int(f_resh[1])), int(r1), (255, 255, 255),
               thickness=5)  # tivknessÊÇ¸ù¾ÝÖÐ¼äµÄÏßÌõÔÚÁ½±ß½øÐÐÑÓÉì×öÏßÌõ´ÖÏ¸µÄ
    cv2.circle(img, (int(f_resh[0]), int(f_resh[1])), int(r2), (255, 255, 255),
               thickness=5)  # thivknessÊÇ¸ù¾ÝÖÐ¼äµÄÏßÌõÔÚÁ½±ß½øÐÐÑÓÉì×öÏßÌõ´ÖÏ¸µÄ
    cv2.imwrite(str_dir, img)