import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.autograd import Variable
import torch
import random

def split_patch(img,patch_size=224,channel=3,step=224):
    '''
    split the image into patches
    :param img: orginal image (PIL Image)
    :param patch_size: default is 224
    :param channel: default is 3
    :param step: default equals to patch_size
    :return:
    img_pad_numpy: the padded image (numpy array)
    patch_arrays: the patches (numpy array)
    '''
    # Image size (W, H)
    w, h = img.size
    img_pad = img.copy()
    p = patch_size
    w_ = w % p
    # padding the image
    if w_ > 0:
        img_pad = F.pad(img_pad, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))

        len_w = w // p + 1
    else:
        len_w = w // p
    h_ = h % p
    if h_ > 0:
        img_pad = F.pad(img_pad, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
        len_h = h // p + 1
    else:
        len_h = h // p
    # transfer the image to array (H,W,C)
    img_pad_numpy = np.array(img_pad)

    # split the image into patches (H//p,W//p,C) ---> (H//p,W//p,p,p,C)
    if channel == 1:
        patch_arrays = patchify(img_pad_numpy,(patch_size,patch_size),step=step)
    else:
        patch_arrays = patchify(img_pad_numpy,(patch_size,patch_size,channel),step=step)

    #img_pad_patch = np.array(img_pad, dtype=np.uint8)
    return img_pad_numpy, patch_arrays

def rebuilt(patch_array,mid_shape,orginal_shape):
    '''
    rebuild the image from patches
    :param patch_array:
    :param mid_shape: shape with padding (H//p,W//p,C)
    :param orginal_shape:
    :return:
    rebuilt_img: the rebuilt image (PIL Image)
    '''
    patch_array = np.array(patch_array)
    # rebuild the image from patches (H//p,W//p,p,p,C) ---> (H,W,C)
    rebuilt_array_mid = unpatchify(patch_array, mid_shape)
    rebuilt_array_mid = np.array(rebuilt_array_mid, dtype=np.uint8)

    # array to image
    rebuilt_img_mid = Image.fromarray(rebuilt_array_mid)

    # crop the image (attention: crop make the shape of the image different from the orginal image)
    crop = transforms.CenterCrop(orginal_shape)
    rebuilt_img = crop(rebuilt_img_mid)

    return rebuilt_img




def patch(img, label, input_size=224):


    w, h = img.size
    w_ = w - input_size
    h_ = h - input_size

    if w_ > 0 and h_ > 0:

        img_crop = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        label_crop = np.zeros((input_size, input_size,3), dtype=np.uint8)
        while np.all(img_crop[:,:,:]==[0,0,0]) or np.all(label_crop[:,:,:]==[0,0,0]):
            img_copy = img.copy()
            label_copy = label.copy()

            x1 = random.choice(range(w_))
            y1 = random.choice(range(h_))

            img_copy = img_copy.crop((x1, y1, x1 + input_size, y1 + input_size))
            img_crop = np.array(img_copy)

            label_copy = label_copy.crop((x1, y1, x1 + input_size, y1 + input_size))
            label_crop = np.array(label_copy)

    return img_copy, label_copy


def patch_one_class(img, label, input_size=224):
    w, h = img.size
    w_ = w - input_size
    h_ = h - input_size

    if w_ > 0 and h_ > 0:
        img_crop = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        label_crop = np.zeros((input_size, input_size), dtype=np.uint8)
        while np.all(img_crop[:,:,:]==[0,0,0]) or np.all(label_crop[:,:]==[0])or np.all(label_crop[:,:]==[1]):
            img_copy = img.copy()
            label_copy = label.copy()

            x1 = random.choice(range(w_))
            y1 = random.choice(range(h_))

            img_copy = img_copy.crop((x1, y1, x1 + input_size, y1 + input_size))
            img_crop = np.array(img_copy)

            label_copy = label_copy.crop((x1, y1, x1 + input_size, y1 + input_size))
            label_crop = np.array(label_copy)

    return img_copy, label_copy

def get_patch_info(shape, p_size):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]
    y = shape[1]
    n = m = 1
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < 5:  # 原本小于50 2022.12.19
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < 5:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)



def global2patch(images, p_size):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''
    patches = []; coordinates = []; templates = []; sizes = []; ratios = [(0, 0)] * len(images); patch_ones = np.ones(p_size)
    for i in range(len(images)):
        w, h = images[i].size
        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        template = np.zeros(size)
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0])
        patches.append([images[i]] * (n_x * n_y))
        coordinates.append([(0, 0)] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1: top = int(np.round(x * step_x))
            else: top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1: left = int(np.round(y * step_y))
                else: left = size[1] - p_size[1]
                template[top:top+p_size[0], left:left+p_size[1]] += patch_ones
                coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])
                patches[i][x * n_y + y] = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1]) #508 508

                # patches[i][x * n_y + y].show()
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)))
    return patches, coordinates, templates, sizes, ratios

def patch2global(patches, n_class, sizes, coordinates, p_size):
    '''
    predicted patches (after classify layer) => predictions
    return: list of np.array
    '''
    patches = np.array(torch.detach(patches).cpu().numpy())
    predictions = [ np.zeros((n_class, size[0], size[1])) for size in sizes]
    for i in range(len(sizes)):
        for j in range(len(coordinates[i])):
            top, left = coordinates[i][j]
            top = int(np.round(top * sizes[i][0])); left = int(np.round(left * sizes[i][1]))
            predictions[i][:, top: top + p_size[0], left: left + p_size[1]] += patches[j][:,:,:]
    return predictions
if __name__ == '__main__':
    print(1)