import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# import pipeline từ project cũ
from attendance_pipeline import FacePipeline

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)
print(app.url_map)

# Khởi tạo pipeline
attendance_pipeline = FacePipeline(
    embeddings_dir="D:/face_recognition_pipeline/data/embeddings",
    model_dir=r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
    spoof_threshold=0.7
)

@app.route('/attendance_embedding', methods=['POST'])
def recognize():

    # # decode base64 → numpy
    # decoded = base64.b64decode(data["image"])
    # nparr = np.frombuffer(decoded, np.uint8)
    # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Lấy file ảnh từ multipart/form-data
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Đọc ảnh bằng OpenCV
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"isSuccess": False, "message": "Invalid image"}), 400



    emb, score, status = attendance_pipeline.recognize(img)
    if emb is None:
        return jsonify({"isSuccess": False, "message": status})
    else:
        return jsonify({"isSuccess": True, "embedding": emb[0].tolist(), "message": status})


@app.route('/')
def index():
    return "Flask app đang chạy thành công!"


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6000)


#dung file tạm để lấy img_path
# import os
# import cv2
# import numpy as np
# import base64
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image

# # import pipeline từ project cũ
# from attendance_pipeline import FacePipeline

# # Khởi tạo Flask app
# app = Flask(__name__)
# CORS(app)
# print(app.url_map)

# # Khởi tạo pipeline
# attendance_pipeline = FacePipeline(
#     embeddings_dir="D:/face_recognition_pipeline/data/embeddings",
#     model_dir=r"D:\face_recognition_pipeline\models\anti_spoofing\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
#     spoof_threshold=0.7
# )

# @app.route('/attendance_embedding', methods=['POST'])
# def recognize():

#     # # decode base64 → numpy
#     # decoded = base64.b64decode(data["image"])
#     # nparr = np.frombuffer(decoded, np.uint8)
#     # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     # Lấy file ảnh từ multipart/form-data
#     file = request.files.get("image")
#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400

#     # Đọc ảnh bằng OpenCV
#     img_bytes = file.read()
#     nparr = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     if img is None:
#         return jsonify({"isSuccess": False, "message": "Invalid image"}), 400

#     tmp_path = "temp.jpg"
#     cv2.imwrite(tmp_path, img)

#     # emb = attendance_pipeline.recognize(tmp_path)
#     emb, score, status = attendance_pipeline.recognize(tmp_path)
#     if emb is None:
#         return jsonify({"isSuccess": False, "message": status})
#     else:
#         return jsonify({"isSuccess": True, "embedding": emb[0].tolist(), "message": status})


# @app.route('/')
# def index():
#     return "Flask app đang chạy thành công!"


# if __name__ == '__main__':
#     app.run(debug=False, host='0.0.0.0', port=6000)

