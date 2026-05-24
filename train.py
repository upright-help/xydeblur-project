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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 구현된 모듈 임포트
from models.unet import BaseUNet
from data.dataset import GoProDataset
from trainer.trainer import Trainer

def main():
    # 1. 하드웨어 세팅
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 시작. 사용 기기: {device}")

    # 2. 데이터 준비 (★주의: img_dir 값을 실제 서버의 데이터 폴더 경로로 변경하세요★)
    train_dataset = GoProDataset(img_dir='./data/GoPro/train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # 3. 모델 초기화 및 적재
    model = BaseUNet().to(device)

    # 4. 훈련 도구 세팅 (논문 기준)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # 5. Trainer 엔진 조립
    trainer = Trainer(model, optimizer, criterion, device)

    # 6. 학습 루프 실행 (테스트용으로 3 에포크만 설정)
    start_epoch = 1
    num_epochs = 5
    
    for epoch in range(start_epoch, num_epochs + 1):
        # 1 에포크 학습 진행
        avg_loss = trainer.train_epoch(train_loader, epoch)
        
        # 스케줄러 업데이트 (학습률 조정)
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"===> Epoch {epoch} 완료 | 평균 Loss: {avg_loss:.6f} | 현재 LR: {current_lr}")

        # (선택) 체크포인트 저장 로직
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), f"checkpoints/unet_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()