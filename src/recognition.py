import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# load Buffalo_M
app = FaceAnalysis(name="buffalo_m", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640,640))
app.prepare(ctx_id=0, det_size=(320,320))

# load embeddings chuẩn
# embeddings_dir = "data/embeddings"
embeddings_dir = r"D:\face_recognition_pipeline\data\embeddings"

embeddings_db = {}
for fname in os.listdir(embeddings_dir):
    if fname.endswith(".npy"):
        person_id = os.path.splitext(fname)[0]
        emb = np.load(os.path.join(embeddings_dir, fname))
        embeddings_db[person_id] = emb

def recognize_face(img_path, threshold=0.5):
    img = cv2.imread(img_path)
    faces = app.get(img)
    if len(faces) == 0:
        print("No face detected:", img_path)
        return None

    emb = faces[0].embedding.reshape(1, -1)

    # so sánh với tất cả vector chuẩn
    best_match = None
    best_score = -1
    for person_id, ref_emb in embeddings_db.items():
        ref_emb = ref_emb.reshape(1, -1)
        score = cosine_similarity(emb, ref_emb)[0][0]
        if score > best_score:
            best_score = score
            best_match = person_id

    if best_score >= threshold:
        print(f"Ảnh {img_path} nhận diện là: {best_match} (similarity={best_score:.3f})")
        return best_match
    else:
        print(f"Ảnh {img_path} không khớp ai (similarity={best_score:.3f})")
        return None

# ví dụ chạy
# test_img = r"C:\Users\huong\Downloads\h6.jpg"
test_img = r"D:\zip\thach\c01ee066-f4e6-44ce-9dc4-323d00ad062c.jpg"

recognize_face(test_img, threshold=0.7)
