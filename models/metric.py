import torch
import torchvision.transforms.functional
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.models.inception import inception_v3
from pytorch_msssim import ssim
import numpy as np
from PIL import Image
from scipy.stats import entropy

def mae(input, target):
    input = (input + 1) / 2
    target = (target + 1) / 2
    range = max(input.max(), target.max()) - min(input.min(), target.min())
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    output = output/range
    return output

def ssim_metric(input, target):
    #cssim_loss = ompare_ssim(input, target)
    input = (input+1)/2
    target = (target+1)/2
    with torch.no_grad():
        ssim_loss = ssim(input, target, data_range= max(input.max(), target.max()) - min(input.min(), target.min()), size_average=True)
    return ssim_loss

def psnr_metric(input, target):
    input = (input + 1) / 2
    target = (target + 1) / 2
    range = max(input.max(), target.max()) - min(input.min(), target.min())
    input = np.array(input.cpu())
    target = np.array(target.cpu())
    _psnr = compare_psnr(input, target, data_range = range)
    return _psnr

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)