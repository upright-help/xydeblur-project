import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(pred, target):
    """ 예측된 이미지와 정답 이미지 사이의 PSNR을 계산합니다. """
    # PyTorch 텐서[C, H, W]를 Numpy 배열[H, W, C]로 변환
    pred = pred.detach().cpu().numpy().transpose(1, 2, 0)
    target = target.detach().cpu().numpy().transpose(1, 2, 0)
    
    # 픽셀 값이 0~1 사이로 정규화되어 있다고 가정 (data_range=1.0)
    return peak_signal_noise_ratio(target, pred, data_range=1.0)

def calculate_ssim(pred, target):
    """ 예측된 이미지와 정답 이미지 사이의 SSIM을 계산합니다. """
    pred = pred.detach().cpu().numpy().transpose(1, 2, 0)
    target = target.detach().cpu().numpy().transpose(1, 2, 0)
    
    return structural_similarity(target, pred, data_range=1.0, channel_axis=-1)