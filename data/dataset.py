import torch.utils.data as data
import torchvision.transforms
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import torch
import random
import numpy as np
import tifffile as tiff
import pandas as pd

import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

import torch.utils.data as data
MASKS = glob.glob('/media/vkon/hdd-data/datasets/test_dataset_rect_freeform/*.png')


# from .util.mask import (bbox2mask, brush_stroke_mask,
#                         get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# mapping from igbp to dfc2020 classes
DFC2020_CLASSES = [
    0,  # class 0 unused in both schemes
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    3,  # --> will be masked if no_savanna == True
    3,  # --> will be masked if no_savanna == True
    4,
    5,
    6,  # 12 --> 6
    7,  # 13 --> 7
    6,  # 14 --> 6
    8,
    9,
    10
    ]

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]
INDEX = [0]

def augment_clear_image(image, percent=30, clear_image_percent=0.2, index=None):
    mask = MASKS[index]
    mask = torchvision.transforms.ToTensor()(Image.open(mask))
    """cover = np.random.rand()
    mask = torch.zeros_like(image)
    if cover < clear_image_percent:
        return image, mask[0:1]
    _, img_height, img_width = image.shape
    total_pixels = img_height * img_width
    cloud_count = np.random.randint(low=0, high=3)
    coverage_percent = np.random.randint(0.25 * percent, percent)
    target_coverage_pixels = int(total_pixels * (coverage_percent / 100))
    for i in range(cloud_count - 1):
        cloud_pix_count = random.randint(10, int(target_coverage_pixels * 0.5))
        cloud_width = int(np.sqrt(cloud_pix_count))
        cloud_height = cloud_width
        target_coverage_pixels -= cloud_width * cloud_height
        target_coverage_pixels = int(target_coverage_pixels)
        pos_y = random.randint(cloud_height, img_height - cloud_height)
        pos_x = random.randint(cloud_width, img_width - cloud_width)
        mask[:, pos_x:pos_x + cloud_width, pos_y:pos_y + cloud_height] = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    cloud_pix_count = target_coverage_pixels
    cloud_width = int(np.sqrt(cloud_pix_count))
    cloud_height = cloud_width
    # print(cloud_height)
    pos_y = random.randint(0, img_height - cloud_height)
    pos_x = random.randint(0, img_width - cloud_width)
    image = image * (1. - mask) + mask * torch.ones_like(image)
    mask[:, pos_x:pos_x + cloud_width, pos_y:pos_y + cloud_height] = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)"""
    #mask = torch.stack([mask, mask, mask])
    #return image, mask[0:1]
    mask = torch.cat([mask, mask, mask])
    #print(mask.max())
    image = image * (1. - mask) + mask * torch.ones_like(image)  # torch.randn_like(image)
    return image, mask[0:1]

def augment_clear_image_random(image, percent=30, clear_image_percent=0.2, index = None):
    mask = MASKS[index]
    mask = torchvision.transforms.ToTensor()(Image.open(mask))
    """cover = np.random.rand()
    mask = torch.zeros_like(image)
    if cover < clear_image_percent:
        return image, mask
    _, img_height, img_width = image.shape
    total_pixels = img_height * img_width
    cloud_count = np.random.randint(low=0, high=3)
    coverage_percent = np.random.randint(0.25 * percent, percent)
    target_coverage_pixels = int(total_pixels * (coverage_percent / 100))
    for i in range(cloud_count - 1):
        cloud_pix_count = random.randint(10, int(target_coverage_pixels * 0.5))
        cloud_width = int(np.sqrt(cloud_pix_count))
        cloud_height = cloud_width
        target_coverage_pixels -= cloud_width * cloud_height
        target_coverage_pixels = int(target_coverage_pixels)
        pos_y = random.randint(cloud_height, img_height - cloud_height)
        pos_x = random.randint(cloud_width, img_width - cloud_width)
        #image[:, pos_x:pos_x + cloud_width, pos_y:pos_y + cloud_height] = np.
        mask[:, pos_x:pos_x + cloud_width, pos_y:pos_y + cloud_height] = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    cloud_pix_count = target_coverage_pixels
    cloud_width = int(np.sqrt(cloud_pix_count))
    cloud_height = cloud_width
    # print(cloud_height)
    pos_y = random.randint(0, img_height - cloud_height)
    pos_x = random.randint(0, img_width - cloud_width)
    mask[:, pos_x:pos_x + cloud_width, pos_y:pos_y + cloud_height] = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    """
    mask = torch.stack([mask, mask, mask])
    image = image*(1.-mask) + mask * torch.ones_like(image)#torch.randn_like(image)
    return image, mask[0:1]
    #return image, mask[0:1]

def free_form_augment(image, lineCount, maxVertex, maxLength, maxBrushWidth, maxAngle, index=None):
    mask = MASKS[index]
    mask = torchvision.transforms.ToTensor()(Image.open(mask))
    """mask = torch.zeros((image.shape[1], image.shape[2]), dtype = torch.uint8)
    lines = np.random.randint(1,lineCount+1)
    for j in range(lines):
        num_vertex = np.random.randint(1, maxVertex + 1)
        start_x = np.random.randint(0, image.shape[0])
        start_y = np.random.randint(0, image.shape[1])
        brush_width = np.random.randint(maxBrushWidth//2, maxBrushWidth)
        pil_mask = Image.fromarray(mask.numpy())
        imagedraw = ImageDraw.Draw(pil_mask)
        for i in range(num_vertex):
            angle = np.random.uniform(0,maxAngle)
            if i%2==0:
                angle = 180- angle
            length = np.random.uniform(0, maxLength)
            end_x = int(start_x  +length  *np.sin(angle * np.pi/180))
            end_y = int(start_y + length * np.cos(angle* np.pi/180))
            imagedraw.line((start_x, start_y, end_x, end_y), fill=255, width=brush_width)
            imagedraw.ellipse((end_x- brush_width//2,end_y - brush_width//2, end_x+ brush_width//2,end_y + brush_width//2),
                              fill = 255)
            start_x = end_x
            start_y = end_y
        mask = torch.tensor(np.array(pil_mask), dtype=torch.uint8)
        flip_code = np.random.randint(-1,2)
        transpose = np.random.randint(0,2)
        if flip_code<2:
            if transpose:
                mask = torch.transpose(mask,0,1)
            mask = torch.flip(mask,(flip_code,))
    mask = mask/255"""
    #mask = torch.stack([mask,mask,mask])
    image = image * (1. - mask) + mask * torch.ones_like(image)#torch.randn_like(image)
    return image, mask[0:1]


def load_s2_normalized(path, use_hr, use_mr, use_lr, index = None):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    s2 = s2.astype(np.float32)
    s2 = torch.tensor(s2)
    if np.random.choice([0,1]):
        s2_cloud, mask = augment_clear_image(s2.clone().detach(), percent=30, clear_image_percent=0.2, index=index)
    else:
        s2_cloud, mask = free_form_augment(s2.clone().detach(), 5, 30, 70, 25, 10, index)
    mean = torch.as_tensor([0.5]*len(bands_selected),
                           dtype=s2.dtype, device=s2.device)
    std = torch.as_tensor([0.5]*len(bands_selected),
                          dtype=s2.dtype, device=s2.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    s2.sub_(mean).div_(std)
    #s2_cloud.sub_(mean).div_(std)

    return s2, s2_cloud, mask

def load_s2_freeform(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    s2 = s2.astype(np.float32)
    s2 = torch.tensor(s2)
    s2_cloud, mask = free_form_augment(s2, 4, 25, 70, 25, 10)
    mean = torch.as_tensor([0.5]*len(bands_selected),
                           dtype=s2.dtype, device=s2.device)
    std = torch.as_tensor([0.5]*len(bands_selected),
                          dtype=s2.dtype, device=s2.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    s2.sub_(mean).div_(std)
    s2_cloud.sub_(mean).div_(std)
    return s2, s2_cloud, mask


# util function for reading s2 data
def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    s2 = s2.astype(np.float32)
    s2 = torch.tensor(s2)
    mean = torch.as_tensor([0.5]*len(bands_selected),
                           dtype=s2.dtype, device=s2.device)
    std = torch.as_tensor([0.5]*len(bands_selected),
                          dtype=s2.dtype, device=s2.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    s2.sub_(mean).div_(std)
    return s2


# util function for reading s1 data
"""def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 /= 25
    s1 += 1
    s1 = s1.astype(np.float32)
    s1 = torch.tensor(s1, dtype = torch.float32)
    mean = torch.as_tensor([0.5, 0.5],
                           dtype=s1.dtype, device=s1.device)
    std = torch.as_tensor([0.5, 0.5],
                          dtype=s1.dtype, device=s1.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    s1.sub_(mean).div_(std)
    return s1"
"""
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    #s1 = np.concatenate([s1,s1[0:1]-s1[1:2]],axis=0)
    s1 = np.clip(s1, -25, 0)
    s1 /= 25
    s1 += 1
    s1 = s1.astype(np.float32)
    s1 = torch.tensor(s1, dtype = torch.float32)
    mean = torch.as_tensor([0.5, 0.5],
                           dtype=s1.dtype, device=s1.device)
    std = torch.as_tensor([0.5, 0.5],
                          dtype=s1.dtype, device=s1.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    s1.sub_(mean).div_(std)
    return s1

def load_s1_normalized(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    #s1 = np.clip(s1, -25, 0)
    #s1 /= 25
    #s1 += 1
    #s1 = s1.astype(np.float32)
    s1 = torch.tensor(s1, dtype = torch.float32)
    mean = s1.mean(dim=0)
    std = torch.as_tensor([0.5, 0.5],
                          dtype=s1.dtype, device=s1.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    s1.sub_(mean).div_(std)
    return s1

# util function for reading lc data
def load_lc(path, no_savanna=False, igbp=True):

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    # convert IGBP to dfc2020 classes
    if igbp:
        lc = np.take(DFC2020_CLASSES, lc)
    else:
        lc = lc.astype(np.int64)

    # adjust class scheme to ignore class savanna
    if no_savanna:
        lc[lc == 3] = 0
        lc[lc > 3] -= 1

    # convert to zero-based labels and set ignore mask
    lc -= 1
    lc[lc == -1] = 255
    return lc


# util function for reading data from single sample
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr, use_s2_cr, use_mask,
                no_savanna=False, igbp=True, unlabeled=False, index = None):

    use_s2 = use_s2hr or use_s2mr or use_s2lr
    #return_dict = {}
    # load s2 data

    if use_s2_cr:
        #return_dict['image_cr'] = load_s2(sample["s2cr"], use_s2hr, use_s2mr, use_s2lr)
        image_cr = load_s2(sample["s2cr"], use_s2hr, use_s2mr, use_s2lr)
    else:
        image_cr = None
    if use_s2:
        #return_dict['image'] = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)
        image, image_cr, mask = load_s2_normalized(sample["s2"], use_s2hr, use_s2mr, use_s2lr, index)
    else:
        image = None
    # load s1 data
    if use_s1:
        #return_dict['image_s1'] = load_s1(sample["s1"])
        image_sar = load_s1(sample["s1"])
    else:
        image_sar = None
    # load label
    #return_dict['id'] = sample["id"]
    if not unlabeled:
        label = load_lc(sample["label"], no_savanna=no_savanna, igbp=igbp)
    else:
        label = None
    return image_cr, image_sar, image, label, mask

# util function for reading data from single sample



# calculate number of input channels
def get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr):
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    return n_inputs


# select channels for preview images
def get_display_channels(use_s2hr, use_s2mr, use_s2lr):
    if use_s2hr and use_s2lr:
        display_channels = [3, 2, 1]
        brightness_factor = 3
    elif use_s2hr:
        display_channels = [0, 1, 2]
        brightness_factor = 3
    elif not (use_s2hr or use_s2mr or use_s2lr):
        display_channels = 0
        brightness_factor = 1
    else:
        display_channels = 0
        brightness_factor = 3
    return (display_channels, brightness_factor)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(
            dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


# calculate number of input channels
def get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr):
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    return n_inputs

def get_ninputs_opt(use_s1, use_s2):
    n_inputs = 0
    if use_s2:
        n_inputs += 3
    if use_s1:
        n_inputs += 3
    return n_inputs

# select channels for preview images
def get_display_channels(use_s2hr, use_s2mr, use_s2lr):
    if use_s2hr and use_s2lr:
        display_channels = [3, 2, 1]
        brightness_factor = 3
    elif use_s2hr:
        display_channels = [2, 1, 0]
        brightness_factor = 3
    elif not (use_s2hr or use_s2mr or use_s2lr):
        display_channels = 0
        brightness_factor = 1
    else:
        display_channels = 0
        brightness_factor = 3
    return (display_channels, brightness_factor)

def get_display_channels_opt(use_s2, use_s1):
    if use_s2:
        display_channels = [0,1,2]
        brightness_factor = 3
    else:
        display_channels = 0
        brightness_factor = 3
    return (display_channels, brightness_factor)

class SyntheticSEN12MS_v3_mask(data.Dataset):
    def __init__(self, path, mode="train", rand_use=0.0, no_savanna=False, use_s2hr=True,
                 use_s2mr=False, use_s2lr=False, use_s2cr=False, use_s1=True, use_mask=False):
        """Initialize the dataset"""

        # inizialize
        super(SyntheticSEN12MS_v3_mask, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        self.use_mask = use_mask
        assert mode in ["train", "val", "test"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
            use_s2hr,
            use_s2mr,
            use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
            self.no_savanna = True
        else:
            self.n_classes = max(DFC2020_CLASSES)
            self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val" or mode == "test":
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer_s2/s2_146":
                broken_file = os.path.join(path, "ROIs1868_summer_s2",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace('_s2',
                                                                                      '_s1')
                self.samples.append({"s2": s2_loc,
                                     "s1": s1_loc,
                                     "id": os.path.basename(s2_loc)})
        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])
        if mode == 'val':
            self.samples = self.samples[:500]
        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        image_cloud, image_sar, image_clear, image_label, image_mask = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_s2cr, self.use_mask, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=True, index = index)
        ret = {}
        ret['gt_image'] = image_clear[[2,1,0],:,:]
        ret['cond_image_sar'] = image_sar
        ret['cond_image'] = torch.cat([image_cloud[[2,1,0],:,:], image_sar, image_mask], dim=0)
        ret['path'] = sample['id']
      #  ret['mask'] = image_mask
        return ret



    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

class SyntheticSEN12MS_v2(data.Dataset):
    def __init__(self, path, mode="train", rand_use=0.0, no_savanna=False, use_s2hr=True,
                 use_s2mr=False, use_s2lr=False, use_s2cr=False, use_s1=True, use_mask=False):
        """Initialize the dataset"""

        # inizialize
        super(SyntheticSEN12MS_v2, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        self.use_mask = use_mask
        assert mode in ["train", "val", "test"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
            use_s2hr,
            use_s2mr,
            use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
            self.no_savanna = True
        else:
            self.n_classes = max(DFC2020_CLASSES)
            self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val" or mode == "test":
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer_s2/s2_146":
                broken_file = os.path.join(path, "ROIs1868_summer_s2",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace('_s2',
                                                                                      '_s1')
                self.samples.append({"s2": s2_loc,
                                     "s1": s1_loc,
                                     "id": os.path.basename(s2_loc)})
        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])
        if mode == 'val':
            self.samples = self.samples[:500]
        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        image_cloud, image_sar, image_clear, image_label, image_mask = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_s2cr, self.use_mask, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=True, index=index)
        ret = {}
        ret['gt_image'] = image_clear[[2,1,0],:,:]
        ret['cond_image_sar'] = image_sar
        ret['cond_image'] = torch.cat([image_cloud[[2,1,0],:,:], image_sar], dim=0)
        ret['path'] = sample['id']
      #  ret['mask'] = image_mask
        return ret



    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

class SyntheticSEN12MS(data.Dataset):
    def __init__(self, path, mode="train", rand_use = 0.0, use_mask=False, no_savanna=False, use_s2hr=True,
                 use_s2mr=False, use_s2lr=False, use_s2cr=False, use_s1=False):
        """Initialize the dataset"""

        # inizialize
        super(SyntheticSEN12MS, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        self.use_mask = use_mask
        assert mode in ["train", "val"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
            self.no_savanna = True
        else:
            self.n_classes = max(DFC2020_CLASSES)
            self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val":
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer_s2/s2_146":
                broken_file = os.path.join(path, "ROIs1868_summer_s2",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)

            for s2_loc in s2_locations:
                self.samples.append({"s2": s2_loc,
                                     "id": os.path.basename(s2_loc)})
        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])
        if mode == 'val':
            self.samples = self.samples[:500]
        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        image_cloud, image_sar, image_clear, image_label, image_mask = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_s2cr, self.use_mask, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=True)
        ret = {}
        ret['gt_image'] = image_clear[[2,1,0],:,:]
        ret['cond_image_sar'] = image_sar
        #image_cloud = self.augment_clear_image(ret['gt_image'].clone().detach())
        ret['cond_image'] = torch.cat([image_cloud[[2,1,0],:,:], image_sar], dim = 0)
        ret['path'] = sample['id']
        if self.use_mask:
            ret['mask'] = image_mask
        return ret



    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

class SyntheticSARSEN12MS(data.Dataset):
    def __init__(self, path, mode="train", rand_use = 0.0, no_savanna=False, use_s2hr=True,
                 use_s2mr=False, use_s2lr=False, use_s2cr=False, use_s1=True, use_mask=False):
        """Initialize the dataset"""

        # inizialize
        super(SyntheticSARSEN12MS, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        self.use_mask = use_mask
        assert mode in ["train", "val", "test"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
            self.no_savanna = True
        else:
            self.n_classes = max(DFC2020_CLASSES)
            self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val" or mode == "test":
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer_s2/s2_146":
                broken_file = os.path.join(path, "ROIs1868_summer_s2",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace('_s2',
                                                                                          '_s1')
                self.samples.append({"s2": s2_loc,
                                     "s1": s1_loc,
                                     "id": os.path.basename(s2_loc)})
        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])
        if mode == 'val':
            self.samples = self.samples[:500]
        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        image_cloud, image_sar, image_clear, image_label, image_mask = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_s2cr, self.use_mask, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=True)
        ret = {}
        ret['gt_image'] = image_clear[[2,1,0],:,:]
        ret['cond_image_sar'] = image_sar[:2]
        #image_cloud = self.augment_clear_image(ret['gt_image'].clone().detach())
        ret['cond_image'] = torch.cat([image_cloud[[2,1,0],:,:], image_sar], dim = 0)
        ret['path'] = sample['id']
        if self.use_mask == True:
            ret['mask'] = image_mask
        return ret



    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


class SyntheticSARSEN12MS_Test(data.Dataset):
    def __init__(self, path, mode="train", rand_use = 0.0, no_savanna=False, use_s2hr=True,
                 use_s2mr=False, use_s2lr=False, use_s2cr=False, use_s1=True, use_mask=False):
        """Initialize the dataset"""

        # inizialize
        super(SyntheticSARSEN12MS, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        self.use_mask = use_mask
        assert mode in ["train", "val", "test"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
            self.no_savanna = True
        else:
            self.n_classes = max(DFC2020_CLASSES)
            self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val" or mode == "test":
            sample_dirs = val_list
        self.masks = glob.glob('/media/vkon/hdd-data/datasets/test_dataset_rect')
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer_s2/s2_146":
                broken_file = os.path.join(path, "ROIs1868_summer_s2",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace('_s2',
                                                                                          '_s1')
                self.samples.append({"s2": s2_loc,
                                     "s1": s1_loc,
                                     "id": os.path.basename(s2_loc)})
        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])
        if mode == 'val':
            self.samples = self.samples[:500]
        assert(len(self.samples)==len(self.masks))
        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        mask = self.masks[index]
        mask = torch.tensor(Image.open(mask))
        image_cloud, image_sar, image_clear, image_label, image_mask = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_s2cr, self.use_mask, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=True)
        mean = torch.as_tensor([0.5] * len(3),
                               dtype=image_clear.dtype, device=image_clear.device)
        std = torch.as_tensor([0.5] * len(3),
                              dtype=image_clear.dtype, device=image_clear.device)
        image_cloud = image * (1. - mask) + mask * torch.ones_like(image)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image_clear.sub_(mean).div_(std)
        ret = {}
        ret['gt_image'] = image_clear[[2,1,0],:,:]
        ret['cond_image_sar'] = image_sar[:2]
        #image_cloud = self.augment_clear_image(ret['gt_image'].clone().detach())
        ret['cond_image'] = torch.cat([image_cloud[[2,1,0],:,:], image_sar], dim = 0)
        ret['path'] = sample['id']
        if self.use_mask == True:
            ret['mask'] = image_mask
        return ret



    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


class SEN12MS(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""
    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val split and can be obtained from:
    #   https://github.com/MSchmitt1984/SEN12MS/blob/master/splits

    def __init__(self, path, mode="train", rand_use = 0.0, no_savanna=False, use_s2hr=True,
                 use_s2mr=False, use_s2lr=False, use_s2cr=True, use_s1=True):
        """Initialize the dataset"""

        # inizialize
        super(SEN12MS, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        assert mode in ["train", "val"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
            self.no_savanna = True
        else:
            self.n_classes = max(DFC2020_CLASSES)
            self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []
        if mode == "train":
            pbar = tqdm(total=162556)   # we expect 541,986 / 3 * 0.9 samples
        else:
            pbar = tqdm(total=18106)   # we expect 541,986 / 3 * 0.1 samples
        pbar.set_description("[Load]")

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val":
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer_s2/s2_146":
                broken_file = os.path.join(path, "ROIs1868_summer_s2",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)
                pbar.write("ignored one sample because of nan values in "
                           + "the s1 data")

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace('_s2',
                                                                                                      '_s1')
                s2cr_loc = s1_loc.replace('_s1', '_s2_cloudy').replace("_s1_", "_s2_cloudy_").replace("s1_", "s2_cloudy_")
                lc_loc = s2_loc.replace('_s2', '_lc').replace("_s2_", "_lc_").replace("s2_", "lc_")

                pbar.update()
                self.samples.append({"label": lc_loc, "s1": s1_loc, "s2": s2_loc, "s2cr": s2cr_loc,
                                     "id": os.path.basename(s2_loc)})

        pbar.close()

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        image_cloud, image_sar, image_clear, image_label = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_s2cr, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=True)
        ret = {}
        ret['gt_image'] = image_clear[[2,1,0],:,:]
        #ret['cond_image_sar'] = image_sar
        ret['cond_image'] = torch.cat([image_cloud[[2,1,0],:,:], image_sar], dim = 0)
        ret['path'] = sample['id']
        return ret


    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

class SEN12MS_Modified(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""
    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val split and can be obtained from:
    #   https://github.com/MSchmitt1984/SEN12MS/blob/master/splits

    def __init__(self, path, mode="train", no_savanna=False, use_s2hr=True,
                 use_s2mr=False, use_s2lr=False, use_s2cr=True, use_s1=True):
        """Initialize the dataset"""

        # inizialize
        super(SEN12MS, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        assert mode in ["train", "val"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
            self.no_savanna = True
        else:
            self.n_classes = max(DFC2020_CLASSES)
            self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []
        if mode == "train":
            pbar = tqdm(total=162556)   # we expect 541,986 / 3 * 0.9 samples
        else:
            pbar = tqdm(total=18106)   # we expect 541,986 / 3 * 0.1 samples
        pbar.set_description("[Load]")

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val":
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer_s2/s2_146":
                broken_file = os.path.join(path, "ROIs1868_summer_s2",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)
                pbar.write("ignored one sample because of nan values in "
                           + "the s1 data")

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace('_s2',
                                                                                                      '_s1')
                s2cr_loc = s1_loc.replace('_s1', '_s2_cloudy').replace("_s1_", "_s2_cloudy_").replace("s1_", "s2_cloudy_")
                lc_loc = s2_loc.replace('_s2', '_lc').replace("_s2_", "_lc_").replace("s2_", "lc_")

                pbar.update()
                self.samples.append({"label": lc_loc, "s1": s1_loc, "s2": s2_loc, "s2cr": s2cr_loc,
                                     "id": os.path.basename(s2_loc)})

        pbar.close()

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        image_cloud, image_sar, image_clear, image_label = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_s2cr, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=True)
        ret = {}
        ret['gt_image'] = image_clear[[2,1,0],:,:]
        #ret['cond_image_sar'] = image_sar
        ret['cond_image'] = torch.cat([image_cloud[[2,1,0],:,:], image_sar], dim = 0)
        ret['path'] = sample['id']
        return ret


    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

class SEN12OPTMS(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""
    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val split and can be obtained from:
    #   https://github.com/MSchmitt1984/SEN12MS/blob/master/splits

    def __init__(self, path, mode="train", rand_use = 0.0, use_s2=True, use_s2cr=True, use_s1=True):
        """Initialize the dataset"""

        # inizialize
        super(SEN12OPTMS, self).__init__()

        # make sure parameters are okay
        if not (use_s2 or use_s2cr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2 = use_s2
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.rand_use = rand_use if mode == "train" else 0.0
        # provide number of input channels
        self.n_inputs = get_ninputs_opt(use_s1, use_s2)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels_opt(
                                                            use_s2, use_s1)

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []
        if mode == "train":
            pbar = tqdm(total=162556)   # we expect 541,986 / 3 * 0.9 samples
        else:
            pbar = tqdm(total=18106)   # we expect 541,986 / 3 * 0.1 samples
        pbar.set_description("[Load]")

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val":
            sample_dirs = val_list[:500]
        elif mode == 'test':
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.png"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == 'ROIs1868_summer_s2/s2_146':
                broken_file = os.path.join(path, "ROIs1868_summer_s2",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.png")
                s2_locations.remove(broken_file)
                pbar.write("ignored one sample because of nan values in "
                           + "the s1 data")

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace('_s2',
                                                                                                      '_s1')
                s2cr_loc = s1_loc.replace('_s1', '_s2_cloudy').replace("_s1_", "_s2_cloudy_").replace("s1_", "s2_cloudy_")
                lc_loc = s2_loc.replace('_s2', '_lc').replace("_s2_", "_lc_").replace("s2_", "lc_")

                pbar.update()
                self.samples.append({"label": lc_loc, "s1": s1_loc, "s2": s2_loc, "s2cr": s2cr_loc,
                                     "id": os.path.basename(s2_loc)})

        pbar.close()

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])
        if mode=='val':
            self.samples = self.samples[:500]
        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)
        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.samples))
        #if self.rand_use:
        #    self.what_to_use = np.random.randint(0,4, len(self.samples))
        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.samples))
        self.augment_flip_param = np.random.randint(0, 3, len(self.samples))
        self.index = 0

    def __getitem__(self, index):
        """Get a single example from the dataset"""
        if self.index >= len(self.samples):
            self.index = 0
        # get and load sample from index file
        sample = self.samples[index]
        image_clear = self.load_sample_opt(sample['s2'])
        if np.random.random_sample() < self.rand_use:
            if np.random.random_sample() < 0.5:
                image_cloud = self.load_sample_opt(sample['s2cr'])
                image_sar = torch.tensor(np.zeros_like(image_cloud, dtype=np.float32))
            else:
                image_sar = self.load_sample_opt(sample['s1'])
                image_cloud = torch.tensor(np.zeros_like(image_sar, dtype = np.float32))
        else:
            image_cloud = self.load_sample_opt(sample['s2cr'])
            image_sar = self.load_sample_opt(sample['s1'])
        ret = {}
        ret['gt_image'] = image_clear
        #ret['cond_image_sar'] = image_sar
        ret['cond_image'] = torch.cat([image_cloud, image_sar], dim = 0)
        ret['path'] = sample['id']
        self.index += 1
        return ret

    def load_sample_opt(self, path):

        img = np.array(Image.open(path)).transpose((2,0,1))
        if self.mode == 'train':
            if not self.augment_flip_param[self.index] == 0:
                img = np.flip(img, self.augment_flip_param[self.index])
            if not self.augment_rotation_param[self.index] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index], (1, 2))


        image = torch.tensor(img.copy())
        image = image / 255.0
        mean = torch.as_tensor([0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image


    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


"""class Sen2_MTC_New_SAR(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []
        files = pd.read_csv(os.path.join(self.data_root))

        if mode == 'train':
            self.tile_list = list(files[])
            #self.tile_list = np.loadtxt(os.path.join(
            #    self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            #self.tile_list = np.loadtxt(
            #    os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            #self.tile_list = np.loadtxt(
             #   os.path.join(self.data_root, 'test.txt'), dtype=str)

        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen12MSCR', tile + '_' + 's2',''))]

            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[
            index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]

        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        # return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, self.image_name[index]
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        ret['cond_image'] = torch.cat([image_cloud0[:3, :, :], image_cloud1[:3, :, :], image_cloud2[:3, :, :]])
        ret['path'] = self.image_name[index] + ".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)[[3,2,1],:,:]
        img = (img / 1.0).transpose((2, 0, 1))

        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index // 4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1

        if self.index // 4 >= len(self.filepair):
            self.index = 0

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image
"""
class Sen2_MTC_New_Multi(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []

        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(
                self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'test.txt'), dtype=str)

        for tile in self.tile_list:
            path = os.path.join(self.data_root, tile)
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(path
                )]
                

            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[
            index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]

        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        # return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, self.image_name[index]
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        ret['cond_image'] = torch.cat([image_cloud0[:3, :, :], image_cloud1[:3, :, :], image_cloud2[:3, :, :]])
        ret['path'] = self.image_name[index]+".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))

        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index//4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1

        if self.index // 4 >= len(self.filepair):
            self.index = 0

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image


class Sen2_MTC_New_Single(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []

        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(
                self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'test.txt'), dtype=str)[:1]

        #self.tile_list_s2 = [x.replace('_s1','_s2_cloudy').replace('s1_','s2_cloudy_') for x in self.tile_list]
        #self.tile_list_s2_cloudless = [x.replace('_s1', '_s2').replace('s1_', 's2_') for x in self.tile_list]
        for tile in self.tile_list:
            tile_s2 =  tile.replace('_s1','_s2_cloudy').replace('s1_','s2_cloudy_')
            tile_s2_cloudless = tile.replace('_s1', '_s2').replace('s1_', 's2_')
            path_s2 = os.path.join(self.data_root, tile_s2)
            #path_cloudless_s2 = os.path.join(self.data_root, tile_s2_cloudless)
            image_name_list_s2 = [image_name.split('.')[0] for image_name in os.listdir(path_s2
                                                                                     )]
            #image_name_list_s2cloudless = [image_name.split('.')[0] for image_name in os.listdir(path_cloudless_s2
                                                                                   #  )]
            for image_name_s2 in image_name_list_s2:
                image_name_s2_cloudless = image_name_s2.replace('_s2_cloudy_', '_s2_')
                image_cloud_path0 = os.path.join(
                    self.data_root, tile_s2, image_name_s2 + '.tif')
                image_cloud_path1 = os.path.join(
                    self.data_root, tile_s2, image_name_s2 + '.tif')
                image_cloud_path2 = os.path.join(
                    self.data_root, tile_s2, image_name_s2 + '.tif')
                image_cloudless_path = os.path.join(
                    self.data_root, tile_s2_cloudless, image_name_s2_cloudless + '.tif')

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name_s2)

        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[
            index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]

        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        # return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, self.image_name[index]
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        ret['cond_image'] = torch.cat([image_cloud0[:3, :, :], image_cloud1[:3, :, :], image_cloud2[:3, :, :]])
        ret['path'] = self.image_name[index] + ".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)[:,:,[3,2,1,7]]
        img = (img / 1.0).transpose((2, 0, 1))

        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index // 4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1

        if self.index // 4 >= len(self.filepair):
            self.index = 0

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image

class Sen2_MTC_New(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []

        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(
                self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'test.txt'), dtype=str)

        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless'))]
                

            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[
            index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]

        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        # return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, self.image_name[index]
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        if self.mode=="train":
            ret['cond_image'] = random.choice([image_cloud0, image_cloud1, image_cloud2])[:3, :, :]
        else:
            ret['cond_image'] = image_cloud0[:3, :, :]
        ret['path'] = self.image_name[index]+".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))

        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index//4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1

        if self.index // 4 >= len(self.filepair):
            self.index = 0

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image

class Sen2_MTC_New2(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []

        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(
                self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(
                os.path.join(self.data_root, 'test.txt'), dtype=str)

        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless'))]
                

            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(
                    self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')

                self.filepair.append(
                    [image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[
            index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]

        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        # return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, self.image_name[index]
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        if self.mode=="train":
            ret['cond_image'] = torch.cat(random.sample((image_cloud0[:3, :, :], image_cloud1[:3, :, :], image_cloud2[:3, :, :]), 2))
        else:
            ret['cond_image'] = torch.cat(image_cloud0[:3, :, :], image_cloud1[:3, :, :])
        ret['path'] = self.image_name[index]+".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))

        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index//4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(
                    img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1

        if self.index // 4 >= len(self.filepair):
            self.index = 0

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5],
                              dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(
                mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0, 2) < 1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(
                    mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(
                    mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader(
            '{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader(
            '{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

def create_rectangle_test_dataset():
    path = '/media/vkon/hdd-data/datasets/test_dataset_rect'
    img = torch.ones((3,256,256))
    for i in range(12666):
        mask = augment_clear_image(img,percent = 30,clear_image_percent=0.0)[1].numpy()
        mask = (mask*255).astype(np.uint8)
        Image.fromarray(mask[0]).save(os.path.join(path, f'{i}.png'))

def create_rectangle_freeform_test_dataset():
    path = '/media/vkon/hdd-data/datasets/test_dataset_rect_freeform'
    img = torch.ones((3,256,256))
    for i in range(12666):
        if np.random.choice([0,1]):
            mask = augment_clear_image(img , percent=30, clear_image_percent=0.0)[1].numpy()
        else:
            mask = free_form_augment(img , 5, 30, 70, 25, 10)[1].numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask[0]).save(os.path.join(path, f'{i}.png'))

def create_reeform_test_dataset():
    path = '/media/vkon/hdd-data/datasets/test_dataset_freeform'
    img = torch.ones((3,256,256))
    for i in range(12666):
        mask = free_form_augment(img , 5, 30, 70, 25, 10)[1].numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask[0]).save(os.path.join(path, f'{i}.png'))

if __name__=='__main__':
    import numpy as np
    create_rectangle_test_dataset()
    create_reeform_test_dataset()
    create_rectangle_freeform_test_dataset()
    #image = torch.zeros((1, 256,256))
    #for i in range(10000):
    #    image, mask = free_form_augment(image, 4, 25, 70, 25, 10)
    #    cv2.imshow("",image.permute(1,2,0).mul_(0.5).add_(0.5).numpy())
    #    cv2.waitKey(0)
