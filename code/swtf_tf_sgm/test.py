import os
import numpy as np
import torch
from swin_unet import SwinUnet
from patch_process import patch2global,global2patch
from PIL import Image
from pandas.core.frame import DataFrame
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

if __name__ == '__main__':
    grade = 'grade1'
    result_log =[]
    img_root = f'/home/yangjy/data/five_class/val/{grade}'
    patch_size = 224
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_path = '/Jane_TF_classification/convnext_bce/code/swtf-tf-sgm/seg_ft_best_weight.pth'
    net = SwinUnet(num_classes=2)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    img_list = os.listdir(img_root)
    for picture in img_list[:1]:
        test_image = os.path.join(img_root,'2006080169_OS.jpg')
        test_image_PIL = Image.open(test_image).convert('RGB')
        patches, coordinates, templates, sizes, ratios = global2patch([test_image_PIL],(patch_size,patch_size))
        net.eval()
        with torch.no_grad():
            predict_patch_list = []
            for patch in patches[0]:
                patch_tensor = transform(patch)
                out_patch = net(torch.unsqueeze(patch_tensor,dim=0))
                predict_patch_list.append(out_patch)
        predict_patch_tensor = torch.concat(predict_patch_list, dim=0)
        results = patch2global(predict_patch_tensor,n_class=2,sizes=sizes,coordinates=coordinates,p_size=(patch_size,patch_size))
        _segment_image_mask_save_sofmax = torch.softmax(torch.tensor(results[0]), dim=0)
        _segment_image_mask_save_max = np.argmax(torch.detach(_segment_image_mask_save_sofmax).numpy(), axis=0)
        save_image_mask_1 = np.zeros((sizes[0][0], sizes[0][1], 3), dtype=np.uint8)
        save_image_mask_1[_segment_image_mask_save_max == 0] = [0, 0, 0]
        save_image_mask_1[_segment_image_mask_save_max == 1] = [255, 255, 255]  # 1 是黑色
        ''''''
        print(np.count_nonzero(_segment_image_mask_save_max == 1),(np.count_nonzero(_segment_image_mask_save_max == 0)+np.count_nonzero(_segment_image_mask_save_max == 1)))
        print(np.count_nonzero(_segment_image_mask_save_max == 1)/(np.count_nonzero(_segment_image_mask_save_max == 0)+np.count_nonzero(_segment_image_mask_save_max == 1)))
        result_log.append([picture,np.count_nonzero(_segment_image_mask_save_max == 1),np.count_nonzero(_segment_image_mask_save_max == 0),np.count_nonzero(_segment_image_mask_save_max == 1)/(np.count_nonzero(_segment_image_mask_save_max == 0)+np.count_nonzero(_segment_image_mask_save_max == 1))])

    print('Finished Training')
    data = DataFrame(data=result_log, columns=['pic', 'tf_num', 'bg_num', 'tf/img'])
    print(data)
    # net = SwinUnet(num_classes=num_classes).to(device)
    #
    # net.eval()
    # with torch.no_grad():
    #     pass
