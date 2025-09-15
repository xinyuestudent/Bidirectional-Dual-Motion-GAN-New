'''
@saulzhang
The implementation code of the evaluation metric in the paper 
"Predicting Future Frames using Retrospective Cycle GAN" with Pytorch
data: Nov,14,2019
'''
import numpy as np
import skimage
import skimage.measure
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def tensor2im(image_numpy, mean=(0.5,0.5,0.5), stddev=0.5):
    image_numpy = (np.transpose(image_numpy,
                                (1, 2, 0)) * stddev + np.array(mean)) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = np.around(image_numpy)
    image_numpy = image_numpy.astype(np.uint8)
    return image_numpy

def MSE(img1,img2):
    img1 = np.transpose(img1,(1, 2, 0)) * 0.5 + np.array(0.5)
    img2 = np.transpose(img2,(1, 2, 0)) * 0.5 + np.array(0.5)
    mse = np.mean((img1 - img2)**2)
    return mse

def MAE(img1,img2):
    img1 = np.transpose(img1,(1, 2, 0)) * 0.5 + np.array(0.5)
    img2 = np.transpose(img2,(1, 2, 0)) * 0.5 + np.array(0.5)
    mae = np.mean(abs(img1 - img2))
    return mae

def PSNR(img1, img2):
    img1 = tensor2im(img1)
    img2 = tensor2im(img2)
    psnr_val = psnr(img1, img2)
    return psnr_val

def SSIM(img1,img2):
    img1 = tensor2im(img1)
    img2 = tensor2im(img2)
    ssim_val = ssim(img1, img2, win_size = 11, data_range = 255, multichannel = True, gaussian_weights = True)
    return ssim_val

def DSSIM(img1,img2):
    img1 = tensor2im(img1)
    img2 = tensor2im(img2)
    ssim_val = ssim(img1, img2, win_size = 11, data_range = 255, multichannel = True, gaussian_weights = True)
    avg_ssim = np.mean(ssim_val)
    dssim = (1-ssim_val)/2
    return dssim


if __name__ == '__main__':

    from PIL import Image
    f1 = Image.open("D:/caltech/caltech/set00_V000/I00002.jpg")
    f2 = Image.open("D:/caltech/caltech/set00_V000/I00005.jpg")
    import torchvision.transforms as transforms
    transforms_ = [
        transforms.Resize((int(256),int(256)), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(transforms_)
    img1 = transform(f1)
    img2 = transform(f2)

    print(SSIM(img1.numpy(),img2.numpy()))