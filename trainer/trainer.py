"""
[trainer.py]
    실제 학습 연산을 수행하는 독립 모듈. 
    외부의 데이터 경로나 하드웨어 설정에 의존하지 않으며, 주입받은 부품들을 이용해 1 에포크(Epoch) 단위의 딥러닝 핵심 연산 로직만을 전담.

필수 구성 요소 (Components):
    1. 초기화 메서드 (__init__): train.py로부터 모델, 옵티마이저, 손실 함수 등을 의존성 주입(Dependency Injection)받아 내부 상태로 저장.
    2. 에포크 실행 메서드 (train_epoch):
        - Data Transfer: 입력 데이터를 연산 디바이스(GPU 등)로 이동.
        - Zero Gradient: 이전 배치의 기울기 찌꺼기 초기화 (optimizer.zero_grad).
        - Forward Pass: 모델을 통과시켜 예측값 산출.
        - Loss Calculation: 정답과 예측값을 비교하여 오차 계산.
        - Backward Pass & Step: 오차를 역전파하여 네트워크 가중치 갱신.
    3. 지표 추적 (Metric Tracking): 배치 단위의 손실값을 누적하여 에포크 평균 손실을 산출 및 반환.
"""
import torch

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader, epoch):
        self.model.train() # 모델을 학습 모드로 설정
        epoch_loss = 0.0

        for batch_idx, (blur, sharp) in enumerate(dataloader):
            # 1. 데이터를 GPU로 이동
            blur, sharp = blur.to(self.device), sharp.to(self.device)

            # 2. 기울기 초기화
            self.optimizer.zero_grad()

            # 3. 순전파
            restored = self.model(blur)

            # 4. 손실 계산
            loss = self.criterion(restored, sharp)

            # 5. 역전파 및 가중치 업데이트
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            # 100번째 배치마다 진행 상황 출력
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(dataloader)}] \t Loss: {loss.item():.6f}")

        # 에포크의 평균 손실 반환
        return epoch_loss / len(dataloader)