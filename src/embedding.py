import os
import numpy as np
from PIL import Image
import cv2
from insightface.app import FaceAnalysis

# Khởi tạo Buffalo_M
app = FaceAnalysis(name="buffalo_m", providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640,640))

#Nếu sau này có GPU, có thể quay lại (640,640) để tăng độ chính xác***
app.prepare(ctx_id=0, det_size=(320,320))

# input_dir = "data/spoof_checked"
# output_dir = "data/embeddings"
input_dir = r"D:\face_recognition_pipeline\data\spoof_checked"
output_dir = r"D:\face_recognition_pipeline\data\embeddings"

os.makedirs(output_dir, exist_ok=True)

for person in os.listdir(input_dir):
    person_dir = os.path.join(input_dir, person)
    embeddings = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print("No face detected:", img_path)
            continue

        # lấy embedding của khuôn mặt đầu tiên
        emb = faces[0].embedding
        embeddings.append(emb)

    if len(embeddings) > 0:
        # trung bình embedding của 3 ảnh
        mean_emb = np.mean(embeddings, axis=0)
        out_path = os.path.join(output_dir, f"{person}.npy")
        np.save(out_path, mean_emb)
        print("Saved embedding:", out_path)
