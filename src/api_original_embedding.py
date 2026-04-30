from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = Flask(__name__)

# Khởi tạo model Buffalo_M
face_app = FaceAnalysis(name="buffalo_m", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320,320))

@app.route("/original_embedding", methods=["POST"])
def get_embedding():
    # data = request.get_json()
    # base64_str = data.get("image")

    # # Giải mã base64 thành ảnh
    # img_bytes = base64.b64decode(base64_str)
    # nparr = np.frombuffer(img_bytes, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Lấy file ảnh từ multipart/form-data
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Đọc ảnh bằng OpenCV
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    faces = face_app.get(img)
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    # Lấy embedding của khuôn mặt đầu tiên
    emb = faces[0].embedding.tolist()

    return jsonify({"embedding": emb})

@app.route('/')
def index():
    return "5000 đang chạy thành công!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
