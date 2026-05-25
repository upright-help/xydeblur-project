"""
[train.py]
    프로젝트의 메인 실행 스크립트. 
    학습에 필요한 모든 부품을 초기화하고 조립 -> 전체 파이프라인을 통제.
    구체적인 수학적 연산 로직은 포함하지 않으며, 거시적인 학습 흐름과 환경을 관리.

필수 구성 요소 (Components):
    1. 환경 설정 (Device Setup): 연산에 사용할 하드웨어(CPU/GPU) 지정.
    2. 데이터 파이프라인 (Data Pipeline): Dataset 객체 생성 및 배치(Batch) 처리를 위한 DataLoader 구성.
    3. 모델 초기화 (Model Instantiation): 학습할 신경망 구조를 메모리에 로드하고 디바이스에 할당.
    4. 훈련 도구 설정 (Optimization Setup): 손실 함수(Criterion), 옵티마이저(Optimizer), 학습률 스케줄러(Scheduler) 정의.
    5. 학습 루프 제어 (Epoch Control): Trainer 객체를 호출하여 지정된 에포크(Epoch)만큼 반복 학습을 지시하고, 결과 로깅 및 모델 가중치(Checkpoint)를 저장.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.unet import BaseUNet, MultiDecoderUNet
from models.xydeblur import XYDeblur
from data.dataset import GoProDataset
from trainer.trainer import Trainer

def get_model(model_name, device):
    """CLI 인자에 따라 알맞은 모델을 반환합니다."""
    if model_name == "unet":
        return BaseUNet().to(device)
    elif model_name == "unet2d":
        return MultiDecoderUNet(num_decoders=2).to(device)
    elif model_name == "unet3d":
        return MultiDecoderUNet(num_decoders=3).to(device)
    elif model_name == "unet4d":
        return MultiDecoderUNet(num_decoders=4).to(device)
    elif model_name == "xydeblur":
        return XYDeblur().to(device)
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

def main():
    # 1. CLI 파서 설정
    parser = argparse.ArgumentParser(description="XYDeblur 훈련 스크립트")
    parser.add_argument('--model', type=str, default='xydeblur', 
                        choices=['unet', 'unet2d', 'unet3d', 'unet4d', 'xydeblur'], 
                        help='훈련할 모델의 이름을 선택하세요.')
    parser.add_argument('--epochs', type=int, default=1300, help='총 훈련 에포크 수 (논문 기준 1300)')
    args = parser.parse_args()

    # 2. 하드웨어 및 데이터 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" [{args.model}] 모델 학습 시작. 사용 기기: {device}")

    train_dataset = GoProDataset(img_dir='./data/train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # 3. 동적 모델 초기화
    model = get_model(args.model, device)

    # 4. 훈련 도구 세팅 (논문 기준 L1 Loss)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    trainer = Trainer(model, optimizer, criterion, device)

    # 체크포인트 저장 폴더 생성
    os.makedirs(f"checkpoints/{args.model}", exist_ok=True)

    # 5. 1300 에포크 학습 루프 실행
    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_epoch(train_loader, epoch)
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"===> Epoch {epoch} 완료 | 평균 Loss: {avg_loss:.6f} | 현재 LR: {current_lr}")

        # 100 에포크 단위로 모델 가중치 저장
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.model}/epoch_{epoch}.pth")

if __name__ == "__main__":
    main()