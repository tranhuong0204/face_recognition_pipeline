import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# load model (ví dụ .pth)
# from SilentFaceModel  import SilentFaceModel  # giả sử bạn có class model
import sys
import os

# thêm đường dẫn src của repo vào sys.path
sys.path.append(r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master")
from src.anti_spoof_predict import AntiSpoofPredict

predictor = AntiSpoofPredict(device_id=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SilentFaceModel()
# model.load_state_dict(torch.load("models/anti_spoofing/silent_face.pth", map_location=device))
# model.eval()

transform = transforms.Compose([
    # transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# input_dir = "data/processed"
# output_dir = "data/spoof_checked"
input_dir = r"D:\face_recognition_pipeline\data\processed"
output_dir = r"D:\face_recognition_pipeline\data\spoof_checked"
os.makedirs(output_dir, exist_ok=True)

threshold = 0.7  # ngưỡng phân biệt thật/giả

for person in os.listdir(input_dir):
    person_dir = os.path.join(input_dir, person)
    output_person_dir = os.path.join(output_dir, person)
    os.makedirs(output_person_dir, exist_ok=True)

    for img_name in os.listdir(person_dir):
        # img_path = os.path.join(person_dir, img_name)
        # img = Image.open(img_path).convert("RGB")
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize((80, 80))
        img_np = np.array(img_resized)[:, :, ::-1]  # RGB → BGR

        tensor = transform(img).unsqueeze(0).to(device)

        # with torch.no_grad():
        #     score = model(tensor).item()  # giả sử output là xác suất spoof

        # result = predictor.predict(img_path)  # trả về score hoặc nhãn

        # chọn model anti-spoofing
        model_path = r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models\4_0_0_80x80_MiniFASNetV1SE.pth"
        img = Image.open(img_path).convert("RGB")
        img = img.resize((80, 80))   # resize về đúng kích thước mà model yêu cầu   # mở file ảnh thành đối tượng PIL.Image
        # result = predictor.predict(img, model_path)
        result = predictor.predict(img_np, model_path)


        score_spoof = float(result[0][1])  # lấy xác suất spoof
        score_real  = float(result[0][0])  # lấy xác suất real
        # # ép kết quả về một số duy nhất (nhìu quá trời quá đất
        # if isinstance(result, torch.Tensor):
        #     score = result.squeeze().detach().cpu().numpy()
        #     score = float(score[0])
        # elif isinstance(result, np.ndarray):
        #     score = float(np.squeeze(result)[0])
        # elif isinstance(result, (list, tuple)):
        #     score = float(result[0])
        # else:
        #     score = float(result)
        if score_spoof < threshold:

            # ảnh thật → lưu lại
            out_path = os.path.join(output_person_dir, img_name)
            img.save(out_path)
            print("Real face:", out_path)
        else:
            print("Spoof detected:", img_path)
