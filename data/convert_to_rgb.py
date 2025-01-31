import os
import pandas as pd
import rasterio
import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
path = '/home/vkon/datasets/SEN12MS-CR/SEN12MSCR'
dest_path = '/home/vkon/datasets/SEN12MS-CR/SEN12OPTCR/'
os.makedirs(dest_path,exist_ok=True)
multi_df_mean = []
multi_df_mean_clipped = []
multi_df_std = []
multi_df_std_clipped = []
multi_df_mean_s2 = []
multi_df_mean_s2_clipped = []
multi_df_std_s2 = []
multi_df_std_s2_clipped = []
s1_df = pd.DataFrame()
for dirpath, _, file_list in os.walk(path):
    pbar = tqdm.tqdm(file_list)
    for file in pbar:
        if file.endswith('.txt'):
            continue
        new_save_path = dirpath.replace(path, dest_path)
        os.makedirs(new_save_path, exist_ok=True)
        path_to_file = os.path.join(dirpath, file)
        save_path = os.path.join(new_save_path, file.replace('.tif','.png'))
        if os.path.exists(save_path):
            continue
        with rasterio.open(path_to_file) as data:
            img = data.read()
        if img.shape[0] == 2:
            mean = img.reshape(2,-1).mean(axis=1)
            std = img.reshape(2,-1).std(axis=1)
            multi_df_mean.append(mean)
            multi_df_std.append(std)
            s1 = img
            s1 = s1.astype(np.float32)
            s1 = np.nan_to_num(s1)
            s1 = np.clip(s1, -25, 0)
            s1 +=25
            s1/=25
            vvvh = s1[0] - s1[1]
            vvvh = np.clip(vvvh,0,1)
            vvvh = vvvh[np.newaxis]
            s1 = np.concatenate([vvvh, s1], axis=0)
            s1*=255
            s1 = s1.transpose([1,2,0]).astype(np.uint8)
            assert cv2.imwrite(save_path, s1)!=-1

        else:
            assert img.shape[0]==13
            mean = img.reshape(13,-1).mean(axis=1)
            std = img.reshape(13,-1).std(axis=1)
            multi_df_mean_s2.append(mean)
            multi_df_std_s2.append(std)
            s2 = img[1:4]
            s2 = s2.astype(np.float32)
            s2 = np.clip(s2, 0, 10000)
            s2 = s2/10000
            s2*=255
            s2 = s2.transpose([1, 2, 0]).astype(np.uint8)
            assert cv2.imwrite(save_path, s2)!=-1





