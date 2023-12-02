# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:25:18 2022

@author: Lenovo
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:25:18 2022

@author: Lenovo
"""

import os
import json
import torch
import torch.optim as optim
from pandas.core.frame import DataFrame
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from model import convnext_tiny as create_model
from five_class_utils import train_one_epoch, evaluate

global log
log = []
import time


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    start_time = time.time()
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(start_time))

    image_path = r'../data/fiveclass-train'
    pretrained_weight = r'../weights/convnext_tiny_1k_224_ema.pth'
    weight_path = rf"../weight/five_class_{start_time}.pth"
    csv_path = rf'../result/five_class_{start_time}.csv'

    num_classes = 5

    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-4
    step_size = 10
    gamma = 0.9

    freeze_layer = True
    early_stop_step = 20
    epochs = 200
    best_acc = 0.5

    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.ColorJitter(0.1, 0.1, 0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # dataset
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    # write class json    {"0": "grade0","1": "grade1","2": "grade2","3": "grade3","4": "grade4"}
    grade_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in grade_list.items())
    json_str = json.dumps(cla_dict, indent=5)
    with open('./five_class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # create model and load weights
    model = create_model(num_classes=num_classes).to(device)
    if 'convnext_tiny_1k_224_ema.pth' in pretrained_weight:
        assert os.path.exists(pretrained_weight), "weights file: '{}' not exist.".format(pretrained_weight)
        weights_dict = torch.load(pretrained_weight, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
        print("Loaded convnext pretrained in ImageNet!")
    elif os.path.exists(pretrained_weight):
        model.load_state_dict(torch.load(pretrained_weight, map_location=device))
        print("Loaded weights pretrained in our data!")
    else:
        print("SORRY!   No pretrained weights!!")

    # freeze layer
    if freeze_layer:
        for name, para in model.named_parameters():
            if ("head" not in name):
                # if ("head" not in name) and ("stages.3" not in name):
                # if ("head" not in name) and ("stages.3" not in name) and ("stages.2" not in name):
                # if ("head" not in name) and ("stages.3" not in name) and ("stages.2" not in name) and ("stages.1" not in name):
                # if ("head" not in name) and ("stages.3" not in name) and ("stages.2" not in name) and ("stages.1" not in name) and ("stages.0" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    else:
        print('No freezing layer!')

    # parameters setting
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # early stop
    total_batch = 0
    last_decrease = 0
    min_loss = 1000

    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, )
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)
        lr_scheduler.step()

        if val_acc > best_acc:  # acc improve save weights
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            print((best_acc, epoch))
            last_decrease = total_batch

        total_batch += 1

        if total_batch - last_decrease > early_stop_step:
            print("No optimization for a long time, auto-stopping...")
            break
        log.append([epoch, train_loss, val_loss, train_acc, val_acc])
    print('Finished Training')
    data = DataFrame(data=log, columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    data.to_csv(csv_path)


if __name__ == '__main__':
    main()
