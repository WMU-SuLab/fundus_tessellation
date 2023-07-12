# -*- coding: utf-8 -*-
import json
import os
import torch
from pandas.core.frame import DataFrame
from torchvision import transforms
from model import convnext_tiny as create_model
from fund_detect.nets import models
from roi_utils import get_fovea_point, judge_grade

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# fd_location model
CHECKPOINT_FILE = "../weight/fd_location.pth"
model_state_dict = torch.load(CHECKPOINT_FILE)
fd_model = models.resnet101(num_classes=2, pretrained=True)
fd_model.load_state_dict(model_state_dict)
fd_model.to(device)

# roi classifier model
model = create_model(num_classes=1).to(device)
weights_root_dir = "../weight"

data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tmp_result = []
i = 0

csv_path = f'../result/RoiCoarseModel.csv'
for grade in ['grade0', 'grade1', 'grade2', 'grade3', 'grade4']:
    img_dir = f'../data/fiveclass-train/val/{grade}'
    # load image
    color_list = os.listdir(img_dir)
    for picture in color_list[:2]:
        i += 1
        print(i)
        img_path = os.path.join(img_dir, picture)
        try:
            isgrade0, prob = judge_grade(device=device, img_path=img_path,
                                         data_transform=data_transform,
                                         fovea=[], grade=0, model=model,
                                         weights_root=weights_root_dir)
            if isgrade0:
                tmp_result.append([picture, 'grade0', grade, prob])
                continue
            else:
                fovea = get_fovea_point(image_path=img_path, fovea_model=fd_model, device=device)
                isgrade4, prob = judge_grade(device=device, img_path=img_path,
                                             data_transform=data_transform,
                                             fovea=fovea, grade=4, model=model,
                                             weights_root=weights_root_dir)
                if isgrade4:
                    tmp_result.append([picture, 'grade4', grade, prob])
                    continue
                else:
                    isgrade3, prob = judge_grade(device=device, img_path=img_path,
                                                 data_transform=data_transform,
                                                 fovea=fovea, grade=3, model=model,
                                                 weights_root=weights_root_dir)
                    if isgrade3:
                        tmp_result.append([picture, 'grade3', grade, prob])
                        continue
                    else:
                        isgrade2, prob = judge_grade(device=device, img_path=img_path,
                                                     data_transform=data_transform,
                                                     fovea=fovea, grade=2, model=model,
                                                     weights_root=weights_root_dir)
                        if isgrade2:
                            tmp_result.append([picture, 'grade2', grade, prob])
                            continue
                        else:
                            tmp_result.append([picture, 'grade1', grade, 1 - prob])
        except Exception as e:
            print(e)
            continue

print(tmp_result)
data = DataFrame(data=tmp_result, columns=['pic_name', 'predict_result', 'label', 'p'])
data.to_csv(csv_path)
print('Finished Predicting')
