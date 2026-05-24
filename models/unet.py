import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 배치 정규화(BN)가 생략된 논문 맞춤형 구조
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class ResModule(nn.Module):
    def __init__(self, channels, num_blocks=6):
        super().__init__()
        # 6개의 ResBlock을 연속으로 연결
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res_module = ResModule(out_channels)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.res_module(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.reduce_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.res_module = ResModule(out_channels)

    def forward(self, x, skip_feature):
        x = self.up_conv(x)
        x = torch.cat([x, skip_feature], dim=1) 
        x = self.reduce_conv(x)
        x = self.res_module(x)
        return x

class Decoder(nn.Module):
    def __init__(self, base_ch=32, out_channels=3):
        super().__init__()
        # [수정 포인트 1] 사라진 1.77M의 주인공인 Bottleneck 전용 ResModule을 추가했습니다.
        self.bottleneck_res = ResModule(base_ch * 4) # 128 채널
        
        # Scale 2 및 Scale 1 복원 블록
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2) # 128 -> 64
        self.dec1 = DecoderBlock(base_ch * 2, base_ch)     # 64 -> 32
        
        # 최종 RGB 출력을 위한 합성곱 층
        self.final_conv = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z, skip2, skip1):
        # [수정 포인트 2] 인코더의 최하단 특징맵 z가 들어오면, 가장 먼저 이 Bottleneck ResModule을 거칩니다.
        x = self.bottleneck_res(z)  
        x = self.dec2(x, skip2)     
        x = self.dec1(x, skip1)     
        return self.final_conv(x)   # 이 디코더 블록이 예측한 잔차(r)를 반환합니다.

class BaseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_ch=32):
        super().__init__()
        
        # 인코더 구조 (기존과 동일)
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.init_res = ResModule(base_ch)
        self.enc1 = EncoderBlock(base_ch, base_ch * 2)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4)
        
        # [수정 포인트 3] 개별 층으로 선언했던 디코더 부품들을 위에서 정의한 'Decoder' 클래스 하나로 대체합니다.
        self.decoder = Decoder(base_ch, out_channels)

    def forward(self, x):
        # 인코더 피처 추출 및 skip-connection용 텐서 저장
        skip1 = self.init_res(self.init_conv(x)) # Scale 1 (32ch)
        skip2 = self.enc1(skip1)                # Scale 2 (64ch)
        z = self.enc2(skip2)                    # Scale 3 (128ch, Bottleneck)
        
        # [수정 포인트 4] 통째로 분리된 디코더에 인코더 특징맵들을 찔러 넣어 잔차를 얻습니다.
        residual = self.decoder(z, skip2, skip1)
        
        # 글로벌 잔차 학습 적용 (입력 이미지 + 디코더 잔차)
        return x + residual

class MultiDecoderUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_ch=32, num_decoders=2):
        super().__init__()
        self.num_decoders = num_decoders
        
        # 인코더 구조 (공유됨 - BaseUNet과 동일)
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.init_res = ResModule(base_ch)
        self.enc1 = EncoderBlock(base_ch, base_ch * 2)
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4)
        
        # [Phase 1 핵심] num_decoders 개수만큼의 디코더를 nn.ModuleList로 묶어서 관리
        self.decoders = nn.ModuleList([
            Decoder(base_ch, out_channels) for _ in range(num_decoders)
        ])

    def forward(self, x):
        # 1. 단일 인코더를 통한 특징(feature) 추출
        skip1 = self.init_res(self.init_conv(x)) # Scale 1
        skip2 = self.enc1(skip1)                 # Scale 2
        z = self.enc2(skip2)                     # Scale 3 (Bottleneck)
        
        # 2. 다중 디코더를 통과시켜 각각의 서브 잔차(sub-residual)를 도출하고 합산
        total_residual = 0
        for decoder in self.decoders:
            total_residual += decoder(z, skip2, skip1)
            
        # 3. 글로벌 잔차 학습 적용 (입력 이미지 + 최종 통합 잔차)
        return x + total_residual
    
# 텐서 크기 및 모델 파라미터 검증 테스트 (기존 __main__ 블록을 대체 또는 수정)
if __name__ == "__main__":
    # 강제로 CPU만 사용하도록 설정
    device = torch.device("cpu") 
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # 1. Base U-Net 테스트
    base_model = BaseUNet().to(device)
    base_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"[Base U-Net] Total Parameters: {base_params / 1e6:.2f}M")
    
    # 2. Multi-Decoder U-Net 파라미터 변화 테스트 (Table 1 검증용)
    for n in [2, 3, 4]:
        multi_model = MultiDecoderUNet(num_decoders=n).to(device)
        multi_params = sum(p.numel() for p in multi_model.parameters() if p.requires_grad)
        output = multi_model(dummy_input)
        
        print(f"[U-Net {n}D] Total Parameters: {multi_params / 1e6:.2f}M | Output shape: {output.shape}")