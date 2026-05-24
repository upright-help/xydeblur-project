import torch
from torch.utils.data import DataLoader
from models.unet import BaseUNet
from data.dataset import GoProDataset
from utils.metrics import calculate_psnr, calculate_ssim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 평가를 시작합니다. 사용 기기: {device}")

    # 1. 평가용 데이터셋 로드 (★ 실제 Test 데이터 경로로 수정 필수 ★)
    test_dataset = GoProDataset(img_dir='./data/GoPro/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. 모델 로드 및 학습된 가중치 불러오기
    model = BaseUNet().to(device)
    # TODO: 실제 학습이 끝난 후, 저장된 가중치 파일 경로로 변경해야 합니다!
    # checkpoint = torch.load('checkpoints/unet_epoch_10.pth')
    # model.load_state_dict(checkpoint)
    model.eval() # 평가 모드 설정 (Dropout, BN 등 비활성화)

    total_psnr = 0.0
    total_ssim = 0.0

    print("채점을 진행 중입니다. 잠시만 기다려주세요...")
    with torch.no_grad(): # 평가할 때는 기울기 계산을 하지 않음 (메모리 절약)
        for batch_idx, (blur, sharp) in enumerate(test_loader):
            blur, sharp = blur.to(device), sharp.to(device)
            
            restored = model(blur)
            
            # 배치 사이즈가 1이므로 0번째 인덱스만 넘겨서 계산
            total_psnr += calculate_psnr(restored[0], sharp[0])
            total_ssim += calculate_ssim(restored[0], sharp[0])

    # 평균 점수 산출
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)

    print(f"최종 평가 결과: PSNR = {avg_psnr:.2f}dB | SSIM = {avg_ssim:.4f}")

if __name__ == "__main__":
    main()