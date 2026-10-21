# Lộ trình thực hiện đề tài 106 — đếm người thời gian thực qua webcam

## Mục tiêu

Xây dựng ứng dụng phát hiện và đếm người trực tiếp từ webcam bằng YOLOv8. Chương trình sẽ hiển thị khung hình từ camera, vẽ bounding box và số người đang xuất hiện trong khung hình hiện tại.

Phạm vi hiện tại không bao gồm nhận diện ảnh tĩnh, video lưu trữ trên máy, đếm người đi qua vạch hoặc ghi nhận tổng lượt người trong cả phiên. Những tính năng đó là phần phát triển tiếp theo.

## Thiết bị và công nghệ

- Python 3.10 hoặc 3.11
- VS Code
- Git và GitHub
- OpenCV
- Ultralytics YOLOv8
- Webcam máy tính hoặc camera được Windows nhận diện như webcam
- iPhone có thể dùng sau nếu muốn thay webcam máy tính, nhưng đây không phải mục tiêu bắt buộc của giai đoạn hiện tại

## Giai đoạn 1 — Khởi tạo repo và môi trường

1. Tạo repository và đặt tên rõ ràng như `yolov8-human-counter`.
2. Khởi tạo Git:

   ```powershell
   git init
   git branch -M main
   ```

3. Tạo `.gitignore` để loại trừ `.venv`, `__pycache__`, file mô hình `.pt` và các output tạm.
4. Tạo môi trường ảo và cài dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Giai đoạn 2 — Cấu trúc thư mục

```text
AI-Human-Detected/
├── ai_human_detected/
│   ├── __init__.py
│   ├── app.py
│   ├── detector.py
│   └── camera.py         # có thể thêm nếu cần tách logic camera
├── tests/
│   └── test_detector.py
├── .gitignore
├── README.md
├── ROADMAP.md
├── main.py
├── requirements.txt
└── outputs/              # tùy chọn, lưu ảnh chụp hoặc video demo
```

Chỉ cần một module chính cho detector và một entrypoint `main.py`. Không tạo không gian phức tạp không cần thiết trong giai đoạn đầu.

## Giai đoạn 3 — Xây dựng detector webcam

1. Dùng OpenCV mở webcam bằng `cv2.VideoCapture(camera_index)`.
2. Chạy YOLO trên từng frame.
3. Chỉ giữ lớp `person` bằng `classes=[0]`.
4. Vẽ bounding box và hiển thị `People: <số lượng>` trên frame.
5. Dừng vòng lặp khi nhấn `q`.
6. Giải phóng camera và đóng cửa sổ khi kết thúc.

Điều kiện thành công: frame hiển thị liên tục, không treo camera, và số người thay đổi theo từng khung hình theo đúng thực tế.

## Giai đoạn 4 — Tích hợp giao diện và thử nghiệm

1. Chạy thử với webcam máy tính.
2. Thử nhiều điều kiện ánh sáng, góc máy và khoảng cách.
3. Đo FPS và tốc độ xử lý.
4. Kiểm tra vấn đề phát hiện nhầm hoặc bỏ sót người.
5. Tránh đặt quá nhiều phụ thuộc vào hiệu ứng UI nếu chưa cần thiết.

## Giai đoạn 5 — Kết nối iPhone nếu cần

Sau khi webcam máy tính hoạt động ổn định, mới xét giai đoạn đánh giá iPhone làm camera thay thế hoặc bổ sung. Cách này thường cần ứng dụng biến iPhone thành webcam trên Windows và kiểm tra với OpenCV.

## Giai đoạn 6 — Đánh giá và báo cáo

1. Ghi lại FPS, số người nhìn thấy theo từng khung hình và mức ổn định của hệ thống.
2. Trích xuất hình ảnh hoặc video demo cho báo cáo.
3. So sánh các cảnh: ít người, đông người, có che khuất, ánh sáng yếu.
4. Viết báo cáo theo cấu trúc: yêu cầu, phương pháp, thiết kế, thực nghiệm, kết quả, hạn chế, hướng phát triển.
5. Hoàn thiện README và outline demo.

## Lộ trình 4 tuần tham khảo

| Tuần | Mục tiêu chính |
|---|---|
| 1 | Khởi tạo repo, cài đặt môi trường, kiểm tra webcam máy tính |
| 2 | Xây dựng YOLO detector trên frame webcam |
| 3 | Tinh chỉnh hiệu năng, thử nhiều cảnh, tối ưu FPS |
| 4 | Viết báo cáo, demo và hướng mở rộng |

## Thứ tự ưu tiên

**Webcam máy tính → YOLO trên frame → đánh giá tốc độ và độ chính xác → iPhone làm nguồn phụ nếu cần → báo cáo.**

Không bắt đầu với ảnh/video lưu sẵn vì mục tiêu chính của đề tài là nhận diện và đếm người theo thời gian thực qua camera.
