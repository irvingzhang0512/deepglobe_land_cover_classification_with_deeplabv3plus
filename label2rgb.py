import numpy as np
import cv2
import scipy.misc
import os
from tqdm import tqdm


def annotation2color(input_path, output_path):
    img = scipy.misc.imread(input_path)

    color = np.zeros((*(2448, 2448), 3), dtype=np.uint8)

    color[img == 0] = [0, 255, 255]
    color[img == 1] = [255, 255, 0]
    color[img == 2] = [255, 0, 255]
    color[img == 3] = [0, 255, 0]
    color[img == 4] = [0, 0, 255]
    color[img == 5] = [255, 255, 255]
    color[img == 6] = [0, 0, 0]

    scipy.misc.imsave(output_path, color)
    pass


one_channel_label_path = '/ssd/zhangyiyang/DeepGlobe/onechannel_label'
test_mask_path = '/ssd/zhangyiyang/DeepGlobe/test_mask'
if not os.path.exists(test_mask_path):
    os.makedirs(test_mask_path)

filelist = os.listdir(one_channel_label_path)
file_names = np.array([file.split('_')[0] for file in filelist if file.endswith('.png')], dtype=object)

for filename in tqdm(file_names):
    label_path = os.path.join(one_channel_label_path, filename + '_label.png')
    mask_path = os.path.join(test_mask_path, filename + '_mask.png')
    annotation2color(label_path, mask_path)
