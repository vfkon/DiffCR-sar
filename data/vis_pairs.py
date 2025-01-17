import os
import pandas
import rasterio
import tqdm
import numpy as np
import matplotlib.pyplot as plt
path = '/home/vkon/datasets/SEN12MS/SEN12MSCR'

for dirpath, _, file_list in os.walk(path):
    pbar = tqdm.tqdm(file_list)
    for file in pbar:
        if file.endswith('.txt'):
            continue
        path_to_file = os.path.join(dirpath, file)
        with rasterio.open(path_to_file) as data:
            img = data.read()
        if img.shape[0]==2:
            continue
        else:
            assert img.shape[0]==13
            mask = img[0]
            mask = mask/mask.max()
            img = img[[3,2,1]]
            img = img/img.max()
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(mask)
            plt.subplot(1,2,2)
            plt.imshow(img.swapaxes(0,2).swapaxes(0,1))
            plt.show()