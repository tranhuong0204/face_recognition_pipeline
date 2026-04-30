import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# import pipeline từ project cũ
from pipeline import FacePipeline

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)
print(app.url_map)

# Khởi tạo pipeline
pipeline = FacePipeline(
    embeddings_dir="D:/face_recognition_pipeline/data/embeddings",
    model_dir=r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
    threshold=0.6,
    spoof_threshold=0.7
)

# @app.route('/verify', methods=['POST'])
# def recognize():
#     data = request.json
#     if not data or "image" not in data:
#         return jsonify({"isSuccess": False, "message": "Thiếu ảnh"}), 400

#     # decode base64 → numpy
#     decoded = base64.b64decode(data["image"])
#     nparr = np.frombuffer(decoded, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # lưu tạm ảnh để pipeline xử lý
#     tmp_path = "temp.jpg"
#     cv2.imwrite(tmp_path, frame)

#     person, score, status = pipeline.recognize(tmp_path)

#     print(f">>> Nhận diện: {person} | Score: {score:.2f} | Status: {status}")

#     return jsonify({
#         "isSuccess": status == "Recognized",
#         "person": person,
#         "score": float(score),
#         "status": status
#     })

@app.route('/verify', methods=['POST'])
def recognize():
    data = request.json
    if not data or "image" not in data or "studentId" not in data:
        return jsonify({"isSuccess": False, "message": "Thiếu dữ liệu"}), 400

    student_id_fe = str(data.get("studentId", ""))

    # decode base64 → numpy
    decoded = base64.b64decode(data["image"])
    nparr = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    tmp_path = "temp.jpg"
    cv2.imwrite(tmp_path, frame)

    person, score, status = pipeline.recognize(tmp_path)

    print(f">>> Nhận diện: {person} | Score: {score:.2f} | Status: {status}")

    # --- So sánh ID ---
    if status == "Recognized" and person is not None:
        import re
        match = re.match(r"([0-9]+)", person)   # lấy phần số đầu tiên từ tên file
        predicted_id = match.group(1) if match else ""

        if predicted_id == student_id_fe:
            return jsonify({
                "isSuccess": True,
                "message": "Điểm danh thành công!",
                "studentId": predicted_id,
                "score": float(score)
            }), 200
        else:
            return jsonify({
                "isSuccess": False,
                "message": f"Sai khuôn mặt! Đây là tài khoản của {student_id_fe}, không phải của bạn.",
                "studentId": predicted_id,
                "score": float(score)
            }), 200
    else:
        return jsonify({
            "isSuccess": False,
            "message": "Không thể xác nhận khuôn mặt chính chủ. Vui lòng chụp lại rõ nét!",
            "score": float(score),
            "status": status
        }), 200

@app.route('/')
def index():
    return "Flask app đang chạy thành công!"


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8088)

# import os
# import cv2
# import numpy as np
# import base64
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity

# import sys
# sys.path.append(r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\src")
# from generate_patches import CropImage
# from utility import parse_model_name

# sys.path.append(r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master")
# from src.anti_spoof_predict import AntiSpoofPredict

# from insightface.app import FaceAnalysis


# # ----------------- Pipeline -----------------
# class FacePipeline:
#     def __init__(self, embeddings_dir, model_dir, threshold=0.5, spoof_threshold=0.7):
#         self.threshold = threshold
#         self.spoof_threshold = spoof_threshold
#         self.embeddings_db = {}
#         for fname in os.listdir(embeddings_dir):
#             if fname.endswith(".npy"):
#                 person_id = os.path.splitext(fname)[0]
#                 emb = np.load(os.path.join(embeddings_dir, fname))
#                 self.embeddings_db[person_id] = emb

#         # load Buffalo_M cho embedding
#         self.app = FaceAnalysis(name="buffalo_m", providers=['CPUExecutionProvider'])
#         self.app.prepare(ctx_id=0, det_size=(320, 320))

#         # load anti-spoofing
#         self.spoof_predictor = AntiSpoofPredict(device_id=0)
#         self.model_dir = model_dir

#     def preprocess(self, img_path):
#         img = Image.open(img_path).convert("RGB")
#         return img

#     def anti_spoof(self, img_path):
#         image = cv2.imread(img_path)
#         image_bbox = self.spoof_predictor.get_bbox(image)
#         image_cropper = CropImage()
#         prediction = np.zeros((1, 3))

#         for model_name in os.listdir(self.model_dir):
#             if not model_name.endswith(".pth"):
#                 continue
#             h_input, w_input, model_type, scale = parse_model_name(model_name)
#             param = {
#                 "org_img": image,
#                 "bbox": image_bbox,
#                 "scale": scale,
#                 "out_w": w_input,
#                 "out_h": h_input,
#                 "crop": True,
#             }
#             if scale is None:
#                 param["crop"] = False
#             img = image_cropper.crop(**param)
#             prediction += self.spoof_predictor.predict(img, os.path.join(self.model_dir, model_name))

#         label = np.argmax(prediction)
#         value = prediction[0][label] / 2.0
#         if label == 1:
#             print(f"Real Face. Score: {value:.2f}")
#             return True
#         else:
#             print(f"Fake Face. Score: {value:.2f}")
#             return False

#     def get_embedding(self, img):
#         img_cv = np.array(img)[:, :, ::-1]  # PIL → BGR
#         faces = self.app.get(img_cv)
#         if len(faces) == 0:
#             return None
#         return faces[0].embedding.reshape(1, -1)

#     def recognize(self, img_path):
#         img = self.preprocess(img_path)

#         if not self.anti_spoof(img_path):
#             return None, 0.0, "Spoof detected"

#         emb = self.get_embedding(img)
#         if emb is None:
#             return None, 0.0, "No face detected"

#         best_match = None
#         best_score = -1
#         for person_id, ref_emb in self.embeddings_db.items():
#             ref_emb = ref_emb.reshape(1, -1)
#             score = cosine_similarity(emb, ref_emb)[0][0]
#             if score > best_score:
#                 best_score = score
#                 best_match = person_id

#         if best_score >= self.threshold:
#             return best_match, best_score, "Recognized"
#         else:
#             return None, best_score, "Unknown"


# # ----------------- Flask API -----------------
# app = Flask(__name__)
# CORS(app)

# pipeline = FacePipeline(
#     embeddings_dir="D:/face_recognition_pipeline/data/embeddings",
#     model_dir=r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
#     threshold=0.5,
#     spoof_threshold=0.7
# )

# @app.route('/')
# def index():
#     return "Flask app đang chạy thành công!"

# @app.route('/verify', methods=['POST'])
# def recognize():
#     data = request.json
#     if not data or "image" not in data:
#         return jsonify({"isSuccess": False, "message": "Thiếu ảnh"}), 400

#     decoded = base64.b64decode(data["image"])
#     nparr = np.frombuffer(decoded, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     tmp_path = "temp.jpg"
#     cv2.imwrite(tmp_path, frame)

#     person, score, status = pipeline.recognize(tmp_path)

#     return jsonify({
#         "isSuccess": status == "Recognized",
#         "person": person,
#         "score": float(score),
#         "status": status
#     })


# if __name__ == '__main__':
#     print(app.url_map)
#     app.run(debug=False, host='0.0.0.0', port=8088)
