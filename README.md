# Đếm người thời gian thực bằng webcam và YOLOv8

Chương trình này tập trung hoàn toàn vào việc phát hiện người trực tiếp từ camera và hiển thị số người đang xuất hiện trong khung hình hiện tại.

Bản hiện tại không xử lý ảnh tĩnh hay video lưu trên máy; mục tiêu là **camera thời gian thực** theo đúng yêu cầu của đề tài.

## Mục tiêu

- Mở webcam trên máy tính.
- Dùng YOLOv8 để phát hiện lớp `person`.
- Hiển thị video, bounding box, số người trong khung hình hiện tại và FPS trên giao diện OpenCV.
- Tạm dừng nhận diện bằng nút trên giao diện hoặc phím `Space`.
- Dừng chương trình bằng nút trên giao diện, phím `q` hoặc `Esc`.
- Làm nền tảng cho các bước nâng cấp sau như: đếm người qua vạch, đếm lượt đi qua, tích hợp iPhone làm webcam.

## Yêu cầu

- Python 3.10 hoặc 3.11
- VS Code
- Webcam máy tính hoặc camera được Windows nhận diện như webcam
- GPU không bắt buộc; `yolov8n.pt` có thể chạy trên CPU

## Cài đặt

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu không kích hoạt được venv, bạn có thể dùng:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy ứng dụng

```powershell
python main.py
```

Nếu máy có nhiều webcam, chọn chỉ số thiết bị:

```powershell
python main.py --camera-index 0
```

Tùy chỉnh ngưỡng tin cậy và kích thước ảnh đầu vào:

```powershell
python main.py --camera-index 0 --confidence 0.35 --image-size 640
```

Các tham số:

- `--camera-index`: chỉ số webcam (0, 1, 2,...)
- `--confidence`: ngưỡng phát hiện, từ 0 đến 1
- `--image-size`: kích thước khung để chạy YOLO
- `--window-name`: tiêu đề cửa sổ hiển thị

## Giao diện và điều khiển

Khi chạy, ứng dụng hiển thị giao diện desktop được dựng bằng OpenCV, gồm vùng video, số người hiện tại, FPS, confidence và image size.

- Click **Pause/Resume** or press `Space` to pause/resume detection. The webcam image continues to update.
- Click **Stop**, or press `q` or `Esc` to exit.
- Use the `-` / `+` buttons in the sidebar to switch cameras. If the selected camera cannot be opened, the app reports an error and keeps the current camera.
- Click the confidence slider or one of the image-size options to change detection settings while the app is running.
- Các tham số `--camera-index`, `--model`, `--confidence` và `--image-size` vẫn có thể cấu hình khi khởi chạy.

The interface uses English labels because OpenCV's built-in Hershey fonts do not support Vietnamese diacritics. Command-line options and this README remain in Vietnamese where appropriate.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

## Kế hoạch triển khai tiếp theo

1. Kiểm tra webcam máy tính hoạt động bằng OpenCV.
2. Chạy module YOLO trên từng khung hình livestream.
3. Đảm bảo số người phát hiện hiển thị ổn định và không giật.
4. Cải thiện tốc độ xử lý nếu FPS thấp.
5. Bổ sung iPhone làm webcam nếu cần nguồn ảnh ngoài máy tính.

## Lưu ý thiết kế

- Chương trình hiện tại nhằm **đếm số người đang xuất hiện trong khung hình hiện tại**, không phải tổng số người đã đi qua trong cả phiên.
- Nếu cần đếm lượt người đi qua, sẽ phát triển thêm một module theo dõi đối tượng hoặc vạch đếm ở giai đoạn sau.
