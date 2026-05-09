import os
import cv2
import numpy as np
from PIL import Image
# from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.append(r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\src")

from generate_patches import CropImage

# ê tại sao phai như này zạ
import sys
sys.path.append(r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\src")
from utility import parse_model_name


# import AntiSpoofPredict từ repo Silent-Face-Anti-Spoofing
import sys
sys.path.append(r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master")
from src.anti_spoof_predict import AntiSpoofPredict


from insightface.app import FaceAnalysis

class FacePipeline:
    def __init__(self, embeddings_dir, model_dir, threshold=0.5, spoof_threshold=0.7):
            self.threshold = threshold
            self.spoof_threshold = spoof_threshold
            self.embeddings_db = {}
            for fname in os.listdir(embeddings_dir):
                if fname.endswith(".npy"):
                    person_id = os.path.splitext(fname)[0]
                    emb = np.load(os.path.join(embeddings_dir, fname))
                    self.embeddings_db[person_id] = emb

            # load Buffalo_M cho embedding
            self.app = FaceAnalysis(name="buffalo_m", providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(320,320))

            # load anti-spoofing
            self.spoof_predictor = AntiSpoofPredict(device_id=0)
            self.model_dir = model_dir   # thư mục chứa nhiều model .pth
            # print(self.app.models.keys())


    # def __init__(self, embeddings_dir, spoof_model_path, threshold=0.5):
    #     self.threshold = threshold
    #     self.embeddings_db = {}
    #     for fname in os.listdir(embeddings_dir):
    #         if fname.endswith(".npy"):
    #             person_id = os.path.splitext(fname)[0]
    #             emb = np.load(os.path.join(embeddings_dir, fname))
    #             self.embeddings_db[person_id] = emb

    #     # load Buffalo_M cho embedding
    #     self.app = FaceAnalysis(name="buffalo_m", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    #     # self.app.prepare(ctx_id=0, det_size=(640,640))
    #     self.app.prepare(ctx_id=0, det_size=(320,320))


    #     # load anti-spoofing
    #     self.spoof_predictor = AntiSpoofPredict(device_id=0)
    #     self.spoof_model_path = spoof_model_path

    def preprocess(self, img_path):
        # đọc ảnh và resize cơ bản
        img = Image.open(img_path).convert("RGB")
        return img


    # def anti_spoof(self, img_path):
    #     img = Image.open(img_path).convert("RGB")
    #     scores = []
    #     for model_name in os.listdir(self.model_dir):
    #         if not model_name.endswith(".pth"):
    #             continue
    #         model_path = os.path.join(self.model_dir, model_name)
    #         h_input, w_input, _, _ = parse_model_name(model_name)
    #         img_resized = img.resize((w_input, h_input))
    #         result = self.spoof_predictor.predict(img_resized, model_path)
    #         score_spoof = float(result[0][1])
    #         scores.append(score_spoof)
    #     avg_spoof = np.mean(scores)
    #     print(f"Average spoof score: {avg_spoof:.3f}")
    #     return avg_spoof < self.spoof_threshold

    # cái trên chưa xử lý giống trong repo gốc nên vẫn còn sai xót

    def anti_spoof(self, img_path):
        image = cv2.imread(img_path)
        image_bbox = self.spoof_predictor.get_bbox(image)
        image_cropper = CropImage()
        prediction = np.zeros((1, 3))

        for model_name in os.listdir(self.model_dir):
            if not model_name.endswith(".pth"):
                continue
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += self.spoof_predictor.predict(img, os.path.join(self.model_dir, model_name))

        label = np.argmax(prediction)
        value = prediction[0][label] / 2.0
        if label == 1:
            print(f"Real Face. Score: {value:.2f}")
            return True
        else:
            print(f"Fake Face. Score: {value:.2f}")
            return False


    def get_embedding(self, img):
        img_cv = np.array(img)[:, :, ::-1]  # PIL → BGR
        faces = self.app.get(img_cv)
        if len(faces) == 0:
            return None
        return faces[0].embedding.reshape(1, -1)

    def recognize(self, img_path):
        img = self.preprocess(img_path)

        # bước anti-spoofing
        if not self.anti_spoof(img_path):
            return None, 0.0, "Spoof detected"

        # bước embedding
        emb = self.get_embedding(img)
        if emb is None:
            return None, 0.0, "No face detected"

        # so sánh với database
        best_match = None
        best_score = -1
        for person_id, ref_emb in self.embeddings_db.items():
            ref_emb = ref_emb.reshape(1, -1)
            score = cosine_similarity(emb, ref_emb)[0][0]
            if score > best_score:
                best_score = score
                best_match = person_id

        if best_score >= self.threshold:
            return best_match, best_score, "Recognized"
        else:
            return None, best_score, "Unknown"

# pipeline = FacePipeline(
#     embeddings_dir="D:/face_recognition_pipeline/data/embeddings",
#     model_dir=r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
#     threshold=0.5,
#     spoof_threshold=0.7
# )

# Load biến môi trường từ file .env
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent  # lên 1 cấp từ src/

pipeline = FacePipeline(
    embeddings_dir=BASE_DIR / os.getenv("EMBEDDINGS_DIR"),
    model_dir=BASE_DIR / os.getenv("MODEL_DIR"),
    threshold=float(os.getenv("THRESHOLD")),
    spoof_threshold=float(os.getenv("SPOOF_THRESHOLD"))
)

test_img = r"C:\Users\huong\Downloads\4026162996690501950.jpg"
person, score, status = pipeline.recognize(test_img)

print(f"Kết quả: {status}, Người: {person}, Similarity={score:.3f}")
