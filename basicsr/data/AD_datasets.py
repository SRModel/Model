import os
import cv2
import random
import math
from basicsr.data.degradations import (
    circular_lowpass_kernel,
    random_mixed_kernels,
)
from basicsr.data.transforms import augment

def generate_low_resolution_image(img_gt, cfg):

    img_gt = augment(img_gt, cfg['use_hflip'], cfg['use_rot'])


    kernel_size = random.choice(cfg['blur_kernel_size'])
    kernel = random_mixed_kernels(
        cfg['kernel_list'],
        cfg['kernel_prob'],
        kernel_size,
        cfg['blur_sigma'],
        cfg['blur_sigma'],
        [-math.pi, math.pi],
        cfg['betag_range'],
        cfg['betap_range'],
        noise_range=None,
    )

    img_gt = cv2.filter2D(img_gt, -1, kernel)


    kernel_size = random.choice(cfg['blur_kernel_size2'])
    kernel = random_mixed_kernels(
        cfg['kernel_list2'],
        cfg['kernel_prob2'],
        kernel_size,
        cfg['blur_sigma2'],
        cfg['blur_sigma2'],
        [-math.pi, math.pi],
        cfg['betag_range2'],
        cfg['betap_range2'],
        noise_range=None,
    )

    lr_img = cv2.filter2D(img_gt, -1, kernel)

    return lr_img

def main():
    cfg = {
        'blur_kernel_size': [21],
        'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'blur_sigma': [0.2, 3],
        'betag_range': [0.5, 4],
        'betap_range': [1, 2],
        'blur_kernel_size2': [21],
        'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'blur_sigma2': [0.2, 1.5],
        'betag_range2': [0.5, 4],
        'betap_range2': [1, 2],
        'use_hflip': True,
        'use_rot': False,
    }

    high_res_folder = '/home/ps/data/data_1/SR'
    output_folder = '/home/ps/data/data_1/SR/DIV2K/'

    for img_name in os.listdir(high_res_folder):
        img_path = os.path.join(high_res_folder, img_name)
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        lr_img = generate_low_resolution_image(img_gt, cfg)
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, lr_img)

if __name__ == '__main__':
    main()
