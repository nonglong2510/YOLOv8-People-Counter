# Lộ trình thực hiện đề tài 106 — đếm người thời gian thực qua webcam

## Mục tiêu

Xây dựng ứng dụng giám sát sức chứa phòng từ webcam bằng YOLOv8. Chương trình hiển thị khung hình, bounding box, số người hiện diện và cảnh báo sức chứa; đồng thời cho phép lưu và xem lại các phiên giám sát.

Số đếm là ước lượng người nhìn thấy trong từng frame, không phải số người duy nhất hay lượt vào/ra. Hệ thống lưu số đếm đã làm mượt theo mẫu thời gian, không lưu hình ảnh hoặc thông tin định danh.

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

## Giai đoạn 4 — Giao diện và cảnh báo sức chứa

1. Hiển thị số người hiện tại, FPS và trạng thái sức chứa.
2. Làm mượt số đếm và ổn định chuyển trạng thái cảnh báo.
3. Cho phép cấu hình giới hạn sức chứa và lưu thiết lập.

## Giai đoạn 5 — Lịch tuần, lưu phiên và báo cáo

1. Người dùng bắt đầu/kết thúc phiên thủ công hoặc cấu hình lịch lặp theo các ngày trong tuần.
2. Tự bắt đầu/kết thúc phiên theo lịch khi ứng dụng và camera đang hoạt động.
3. Lưu mẫu số người đã làm mượt mỗi giây vào SQLite; không lưu ảnh.
4. Xem lại năm phiên gần nhất và xuất báo cáo tổng hợp CSV.
5. Thời lượng trạng thái được ước lượng từ các mẫu quan sát; thời gian tạm dừng không được ghi mẫu.

## Giai đoạn 6 — Đánh giá, hoàn thiện và báo cáo

1. Đo sai số đếm và FPS trong các cảnh ít người, đông người, che khuất và ánh sáng yếu.
2. Ghi rõ dữ liệu thử nghiệm, giới hạn và điều kiện camera; không suy diễn độ chính xác khi chưa đo.
3. Hoàn thiện sơ đồ Use Case, ERD, CRUD và sơ đồ tuần tự theo chức năng đã triển khai.
4. Viết báo cáo, chuẩn bị video demo và cập nhật tài liệu.

## Hướng phát triển tùy chọn

Lọc báo cáo theo ngày/tuần/tháng, quản lý nhiều phòng/camera, tích hợp camera iPhone hoặc đếm lượt qua vạch chỉ nên thực hiện nếu nằm trong phạm vi đồ án.
