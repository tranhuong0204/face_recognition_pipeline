import os
from PIL import Image
import numpy as np
import sys

sys.path.append(r"D:\zip\Silent-Face-Anti-Spoofing-master")
from src.anti_spoof_predict import AntiSpoofPredict

import sys
sys.path.append(r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\src")

from utility import parse_model_name

model_dir = r"D:\zip\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models"
test_img = r"C:\Users\huong\Downloads\h11a.jpg"

predictor = AntiSpoofPredict(device_id=0)
img = Image.open(test_img).convert("RGB")

scores = []
for model_name in os.listdir(model_dir):
    if not model_name.endswith(".pth"):
        continue
    model_path = os.path.join(model_dir, model_name)
    h_input, w_input, _, _ = parse_model_name(model_name)
    img_resized = img.resize((w_input, h_input))
    result = predictor.predict(img_resized, model_path)
    score_spoof = float(result[0][1])
    scores.append(score_spoof)
    print(f"{model_name}: spoof={score_spoof:.3f}")

avg_spoof = np.mean(scores)
print(f"Average spoof score: {avg_spoof:.3f}")
