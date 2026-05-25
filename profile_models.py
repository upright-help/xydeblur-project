import torch
from thop import profile

# 구현된 모델들 임포트
from models.unet import BaseUNet, MultiDecoderUNet
from models.xydeblur import XYDeblur

def main():
    print("모델 프로파일링을 시작합니다... (CPU 환경에서 연산 중)")
    device = torch.device("cpu") # 고해상도 텐서로 인한 GPU OOM(메모리 부족) 방지를 위해 CPU 사용
    
    # 논문 평가 기준인 720P 해상도 입력: (Batch=1, Channels=3, Height=720, Width=1280)
    dummy_input = torch.randn(1, 3, 720, 1280).to(device)
    
    # 평가할 5개 모델 딕셔너리 구성
    models_to_test = {
        "U-Net": BaseUNet().to(device),
        "U-Net^2D": MultiDecoderUNet(num_decoders=2).to(device),
        "U-Net^3D": MultiDecoderUNet(num_decoders=3).to(device),
        "U-Net^4D": MultiDecoderUNet(num_decoders=4).to(device),
        "XYDeblur": XYDeblur().to(device)
    }
    
    print("-" * 55)
    print(f"{'Model Name':<15} | {'Params (M)':<15} | {'GMACs':<15}")
    print("-" * 55)
    
    for name, model in models_to_test.items():
        # thop.profile을 통해 MACs(연산 횟수)와 Params(파라미터 수) 추출
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        
        # 단위 변환: 1 GMAC = 10^9 MACs, 1 M Params = 10^6 Params
        macs_g = macs / 1e9
        params_m = params / 1e6
        
        print(f"{name:<15} | {params_m:>10.2f} M | {macs_g:>10.2f} G")
        
    print("-" * 55)
    print("측정이 완료되었습니다.")

if __name__ == "__main__":
    main()