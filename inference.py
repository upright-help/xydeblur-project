import torch
import torchvision.transforms as transforms
from PIL import Image
from models.unet import BaseUNet

def infer_single_image(model_path, image_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 준비
    model = BaseUNet().to(device)
    # model.load_state_dict(torch.load(model_path)) # 나중에 주석 해제!
    model.eval()

    # 2. 이미지 불러오기 및 텐서 변환
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device) # [1, C, H, W] 형태로 만들기

    # 3. 모델 통과 (복원)
    with torch.no_grad():
        restored_tensor = model(input_tensor)

    # 4. 결과 텐서를 다시 이미지로 변환해서 저장
    # 값을 0~1 사이로 잘라주고(clamp) 형태 변환
    restored_tensor = restored_tensor.squeeze(0).clamp(0, 1) 
    to_pil = transforms.ToPILImage()
    restored_img = to_pil(restored_tensor)
    
    restored_img.save(save_path)
    print(f"복원 완료 '{save_path}' 파일이 저장되었습니다.")

if __name__ == "__main__":
    # ★ 나중에 학습이 다 끝나면 사용할 경로들입니다. 지금 당장 실행할 필요는 없습니다. ★
    # infer_single_image(
    #     model_path="checkpoints/unet_epoch_10.pth", 
    #     image_path="./data/test_blur.png", 
    #     save_path="./restored_result.png"
    # )
    print("Inference 스크립트가 준비되었습니다.")