# -*- coding: utf-8 -*-
import json
from PIL import Image
from torchvision import transforms
from model import convnext_tiny as create_model
from pandas.core.frame import DataFrame
import torch
import os

log = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize([0.456, 0.485, 0.406], [0.224, 0.229, 0.225])])

num_classes = 1

grade = 'grade1'
pretrained_weight_path = rf'../weight/{grade}_part.pth'
save_predict_path = rf'../result/{grade}_predict.csv'

# read class_indict
json_path = r'./roi_class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

model = create_model(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(pretrained_weight_path, map_location=device))

i = 0
img_dir = rf'../data/fivepart-train/{grade}/val/yes'
color_list = os.listdir(img_dir)

for picture in color_list:
    i += 1
    print(i)
    img_path = os.path.join(img_dir, picture)
    try:
        img = Image.open(img_path)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        res = [picture, class_indict[str(predict_cla)], grade, torch.max(predict).numpy()]
        log.append(res)


    except Exception as e:
        print(e)
        continue

data = DataFrame(data=log, columns=['pic_name', 'predict_result', 'label', 'p'])
data.to_csv(save_predict_path)
print('Finished Predicting')
