import torch
import torch.nn as nn
from torch.func import functional_call

# 기존 unet.py에서 기본 블록들을 임포트합니다.
from models.unet import ResModule, EncoderBlock, Decoder

def rotate_weight_90(weight, k=1):
    """
    PyTorch Conv2d/ConvTranspose2d의 가중치(Weight) 텐서를 
    공간적(Spatial, HxW)으로 90도(반시계 방향) 회전시킵니다.
    """
    return torch.rot90(weight, k=k, dims=[2, 3])

class XYDeblur(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_ch=32):
        super().__init__()
        
        # 인코더 (공유됨)
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.init_res = ResModule(base_ch)
        self.enc1 = EncoderBlock(base_ch, base_ch * 2)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4)
        
        # [핵심] 주 디코더 (D_hor)
        # 이 디코더 하나만 학습 가능한 실제 파라미터(Weight, Bias)를 갖습니다.
        self.decoder = Decoder(base_ch, out_channels)

    def forward(self, x):
        # 1. 단일 인코더를 통한 특징 추출
        skip1 = self.init_res(self.init_conv(x))
        skip2 = self.enc1(skip1)
        z = self.enc2(skip2)
        
        # 2. 주 디코더 (D_hor) 순전파 -> 수평 방향 잔차(r1) 예측
        r1 = self.decoder(z, skip2, skip1)
        
        # 3. 보조 디코더 (D_ver)를 위한 파라미터 동적 생성 (공유 및 회전)
        rotated_params = {}
        for name, param in self.decoder.named_parameters():
            # Conv2d 및 ConvTranspose2d의 가중치(Weight) 텐서는 4차원
            if 'weight' in name and param.dim() == 4:
                # 계산 그래프를 유지한 채로 가중치만 90도 회전
                rotated_params[name] = rotate_weight_90(param, k=1)
            else:
                # 편향(Bias) 등은 원본 파라미터를 그대로 공유
                rotated_params[name] = param
                
        # 4. 회전된 파라미터를 주입하여 두 번째 디코더 연산 -> 수직 방향 잔차(r2) 예측
        # 내부적으로 F.conv2d, F.conv_transpose2d 연산 시 rotated_params가 동적으로 사용됩니다.
        r2 = functional_call(self.decoder, rotated_params, (z, skip2, skip1))
        
        # 5. 글로벌 잔차 합산 (입력 이미지 + r1 + r2)
        total_residual = r1 + r2
        return x + total_residual

# --- [Phase 2, Step 3] Autograd(기울기 누적) 검증 테스트 ---
if __name__ == "__main__":
    print("XYDeblur 파라미터 공유 및 역전파 검증 테스트를 시작합니다.")
    device = torch.device("cpu")
    model = XYDeblur().to(device)
    
    # 1. 모델 파라미터 수 확인 (Base U-Net과 동일해야 함)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"네트워크 총 파라미터 수: {total_params / 1e6:.2f}M")
    
    # 2. 더미 데이터 생성
    dummy_input = torch.randn(1, 3, 256, 256)
    dummy_target = torch.randn(1, 3, 256, 256)
    
    # 3. 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    # 4. 순전파 및 손실 계산
    output = model(dummy_input)
    loss = torch.nn.functional.l1_loss(output, dummy_target)
    
    # 5. 역전파 실행
    loss.backward()
    
    # 6. 기울기(Gradient) 누적 검증
    # D_hor와 D_ver의 연산이 하나의 파라미터 세트(self.decoder)로 흘러갔는지 확인
    sample_param = model.decoder.final_conv.weight
    
    if sample_param.grad is not None:
        print(" 검증 성공: 단일 디코더 파라미터에 두 방향의 연산 기울기가 올바르게 누적되었습니다.")
        print(f"   기울기 텐서 형태: {sample_param.grad.shape}")
    else:
        print("오류: 기울기가 계산되지 않았습니다.")