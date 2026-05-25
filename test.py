import argparse
import os
import glob
import torch
import re
from torch.utils.data import DataLoader

from models.unet import BaseUNet, MultiDecoderUNet
from models.xydeblur import XYDeblur
from data.dataset import GoProDataset
from utils.metrics import calculate_psnr, calculate_ssim


def get_model(model_name, device):
    if model_name == "unet": return BaseUNet().to(device)
    elif model_name == "unet2d": return MultiDecoderUNet(num_decoders=2).to(device)
    elif model_name == "unet3d": return MultiDecoderUNet(num_decoders=3).to(device)
    elif model_name == "unet4d": return MultiDecoderUNet(num_decoders=4).to(device)
    elif model_name == "xydeblur": return XYDeblur().to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'unet2d', 'unet3d', 'unet4d', 'xydeblur'])
    parser.add_argument('--ckpt_dir', type=str, default=None, help='평가할 체크포인트 폴더 경로')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else f"checkpoints/{args.model}"
    
    # 해당 모델의 모든 가중치 파일 검색 (.pth)
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")), 
                    key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0] if re.findall(r'\d+', os.path.basename(x)) else 0))
    
    if not ckpt_files:
        print(f"{ckpt_dir} 경로에 평가할 가중치 파일이 없습니다.")
        return

    test_dataset = GoProDataset(img_dir='./data/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = get_model(args.model, device)
    
    print(f"[{args.model}] 일괄 평가를 시작합니다. 총 {len(ckpt_files)}개의 체크포인트 발견.")

    for ckpt in ckpt_files:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        total_psnr, total_ssim = 0.0, 0.0

        with torch.no_grad():
            for blur, sharp in test_loader:
                blur, sharp = blur.to(device), sharp.to(device)
                restored = model(blur)
                total_psnr += calculate_psnr(restored[0], sharp[0])
                total_ssim += calculate_ssim(restored[0], sharp[0])

        avg_psnr = total_psnr / len(test_loader)
        avg_ssim = total_ssim / len(test_loader)
        print(f"[{os.path.basename(ckpt)}] 결과: PSNR = {avg_psnr:.2f}dB | SSIM = {avg_ssim:.4f}")

if __name__ == "__main__":
    main()