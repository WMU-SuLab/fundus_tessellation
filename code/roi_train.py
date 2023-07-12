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
from torchvision import transforms, datasets
from model import convnext_tiny as create_model
from roi_utils import get_params_groups, train_one_epoch, evaluate
import time
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    start_time = time.time()
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(start_time))

    grade = 'grade1'
    image_path = f'../data/fivepart-train/{grade}'
    pretrained_weight = '../weight/convnext_tiny_1k_224_ema.pth'
    weight_path = f"../weight/{grade}_{start_time}.pth"
    csv_path = f'../result/{grade}_{start_time}.csv'
    log = []

    num_classes = 1
    batch_size = 32
    freeze_layers = True

    learning_rate = 1e-4
    weight_decay = 1e-3

    step_size = 10
    gamma = 0.5

    early_stop_step = 10

    epochs = 200
    best_acc = 0.5

    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.ColorJitter(0.5, 0.2, 0.2),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)

    grade_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in grade_list.items())
    json_str = json.dumps(cla_dict, indent=num_classes)
    with open('./roi_class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model = create_model(num_classes=1).to(device)

    if 'convnext_tiny_1k_224_ema.pth' in pretrained_weight:
        assert os.path.exists(pretrained_weight), "weights file: '{}' not exist.".format(pretrained_weight)
        weights_dict = torch.load(pretrained_weight, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
        print("Loaded convnext pretrained in ImageNet!")

    elif pretrained_weight != "":
        assert os.path.exists(pretrained_weight), "weights file: '{}' not exist.".format(pretrained_weight)
        model.load_state_dict(torch.load(pretrained_weight, map_location=device))
        print("Loaded weight pretrained in our data!")
    else:
        print("SORRY!   No pretrained weight!!")

    if freeze_layers == True:
        for name, para in model.named_parameters():
            if ("head" not in name):
                # if ("head" not in name) and ("stages.3" not in name):
                # if ("head" not in name) and ("stages.3" not in name) and ("stages.2" not in name):
                # if ("head" not in name) and ("stages.3" not in name) and ("stages.2" not in name) and ("stages.1" not in name):
                # if ("head" not in name) and ("stages.3" not in name) and ("stages.2" not in name) and ("stages.1" not in name) and ("stages.0" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = get_params_groups(model, weight_decay=weight_decay)
    optimizer = optim.AdamW(pg, lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    total_batch = 0
    last_decrease = 0

    for epoch in range(epochs):
        # train.py
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                )
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)
        lr_scheduler.step()

        log.append([epoch, train_loss, val_loss, train_acc, val_acc])
        if val_acc > best_acc:  # acc improve save weight
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            last_decrease = total_batch
            print((best_acc, last_decrease))
        total_batch += 1

        if total_batch - last_decrease > early_stop_step:
            print("No optimization for a long time, auto-stopping...")
            break
    print('Finished Training')
    data = DataFrame(data=log, columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    data.to_csv(csv_path)


if __name__ == '__main__':
    main()
