# Face Recognition Pipeline

## Setup

1. Tạo môi trường ảo và cài dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

   pip install -r requirements.txt

2. InsightFace sẽ tự động tải model buffalo_m khi chạy lần đầu.
    Ví dụ:
    python

    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_m")
    app.prepare(ctx_id=0)

    → Model sẽ được tải về cache (~/.insightface/models/) và dùng ngay.

3. Tải từ repo gốc: Silent-Face-Anti-Spoofing (github.com in Bing)

    Đặt file model vào thư mục:
    
    models/anti_spoofing/