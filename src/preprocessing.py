from mtcnn import MTCNN
import cv2
import os
from PIL import Image

input_dir = "data/raw"
output_dir = "data/processed"
image_size = 112
margin = 10

detector = MTCNN()

for person in os.listdir(input_dir):
    person_dir = os.path.join(input_dir, person)
    output_person_dir = os.path.join(output_dir, person)
    os.makedirs(output_person_dir, exist_ok=True)

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(img_rgb)
        if len(faces) == 0:
            print("No face:", img_path)
            continue

        # lấy mặt lớn nhất
        face = max(faces, key=lambda f: f['box'][2]*f['box'][3])
        x, y, w, h = face['box']
        x1 = max(x - margin//2, 0)
        y1 = max(y - margin//2, 0)
        x2 = min(x+w + margin//2, img_rgb.shape[1])
        y2 = min(y+h + margin//2, img_rgb.shape[0])

        cropped = img_rgb[y1:y2, x1:x2]
        cropped = Image.fromarray(cropped)
        scaled = cropped.resize((image_size, image_size), Image.ANTIALIAS)

        out_path = os.path.join(output_person_dir, os.path.splitext(img_name)[0] + ".png")
        scaled.save(out_path)
        print("Aligned:", out_path)
