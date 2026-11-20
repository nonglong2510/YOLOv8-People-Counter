# Đếm người thời gian thực bằng webcam và YOLOv8

Chương trình này tập trung hoàn toàn vào việc phát hiện người trực tiếp từ camera và hiển thị số người đang xuất hiện trong khung hình hiện tại.

Bản hiện tại không xử lý ảnh tĩnh hay video lưu trên máy; mục tiêu là **camera thời gian thực** theo đúng yêu cầu của đề tài.

## Mục tiêu

- Mở webcam trên máy tính.
- Dùng YOLOv8 để phát hiện lớp `person`.
- Hiển thị video, bounding box, số người trong khung hình hiện tại và FPS trên giao diện OpenCV.
- So sánh số người trong frame với sức chứa phòng và hiển thị trạng thái bình thường, gần ngưỡng, đạt ngưỡng hoặc vượt ngưỡng.
- Ghi nhận phiên giám sát, lưu mẫu số người đã làm mượt mỗi giây và xem lại các phiên gần đây.
- Tổng hợp số người trung bình/cao nhất và thời lượng quan sát ở trạng thái gần/đạt/vượt sức chứa; xuất báo cáo CSV.
- Tạo lịch lặp theo các ngày trong tuần để tự động bắt đầu/kết thúc phiên giám sát.
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
python main.py --camera-index 0 --confidence 0.35 --image-size 640 --capacity-limit 30
```

Các tham số:

- `--camera-index`: chỉ số webcam (0, 1, 2,...)
- `--confidence`: ngưỡng phát hiện, từ 0 đến 1
- `--image-size`: kích thước khung để chạy YOLO
- `--capacity-limit`: sức chứa tối đa phòng (mặc định 30)
- `--window-name`: tiêu đề cửa sổ hiển thị

## Giao diện và điều khiển

Khi chạy, ứng dụng hiển thị giao diện desktop được dựng bằng OpenCV, gồm vùng video, số người hiện tại, FPS, confidence và image size.

- Click **Pause/Resume** or press `Space` to pause/resume detection. The webcam image continues to update.
- Click **Stop**, or press `q` or `Esc` to exit.
- Use the `-` / `+` buttons in the sidebar to switch cameras. If the selected camera cannot be opened, the app reports an error and keeps the current camera.
- Click the confidence slider or one of the image-size options to change detection settings while the app is running.
- Use the `-` / `+` buttons in the Room Capacity card to adjust the maximum room capacity. The default is 30 people; “Near Capacity” starts at 80% of the limit.
- Click **Save settings** to store the current camera, model, confidence, image size, and room capacity for the next run.
- Click **Defaults** to restore the default camera, model, confidence, image size, and room capacity. Restart the app to load the default model if a custom model was used.
- Click **Start session** to begin saving a smoothed occupancy sample once per second; click **End session** to finish and summarize it. An active session is also ended when the application exits.
- Room capacity is fixed for the duration of a session so its alerts and report use a consistent limit.
- Click **History** to show the five latest sessions, or **Export CSV** to write a summary of all sessions.
- Click **Schedules** to add, edit, enable, disable, or delete weekly schedules. Select weekdays, then adjust start/end time in 15-minute steps and click **Save**. The default draft is daily from 07:00 to 09:30; it is not enabled until saved as enabled.
- An enabled schedule starts and ends sessions automatically using the computer's local time. If the app opens during an active schedule, it starts that session after the webcam opens. The app and camera must remain running; this feature does not launch the app when Windows starts.
- When a scheduled shift starts while a manual session is running, the manual session is closed and the scheduled session takes over.
- Weekly schedules cannot cross midnight or overlap another enabled schedule on the same weekday. Monthly calendar schedules are not supported yet.
- Samples are not recorded while detection is paused. Session durations in reports are estimated from observed samples; gaps longer than one second are excluded.
- Settings are stored in the user's application configuration folder (`%APPDATA%\AIHumanDetected\settings.json` on Windows).
- Session data is stored in `%APPDATA%\AIHumanDetected\sessions.sqlite3`; the CSV report is saved beside it as `session-reports.csv`.
- Command-line options `--camera-index`, `--model`, `--confidence`, `--image-size`, and `--capacity-limit` override saved values for the current run.

The interface uses English labels because OpenCV's built-in Hershey fonts do not support Vietnamese diacritics. Command-line options and this README remain in Vietnamese where appropriate.

Capacity status is based on a rolling median of the last five detected frame counts and changes after three consecutive frames confirm the new status. States are Normal, Near Capacity (at least 80% of the limit), At Capacity, and Over Capacity. The displayed current-frame count remains the direct model result; the capacity card shows the smoothed estimate. This estimates visible people; it does not count attendance or unique visitors.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

## Hạn chế và hướng phát triển tiếp theo

1. Đánh giá định lượng số đếm và FPS trong nhiều điều kiện ánh sáng, khoảng cách và che khuất.
2. Bổ sung lọc báo cáo theo ngày/tuần/tháng nếu cần cho phạm vi đồ án.
3. Cân nhắc nhiều phòng/camera và cấu hình tên phòng.

## Lưu ý thiết kế

- Số người hiện tại là **ước lượng số người nhìn thấy trong khung hình**, không phải điểm danh hoặc số người duy nhất.
- Trạng thái gần ngưỡng bắt đầu từ 80% sức chứa; đạt ngưỡng và vượt ngưỡng được hiển thị riêng.
- Kết quả có thể bị ảnh hưởng bởi ánh sáng, góc camera và việc người bị che khuất.
- Tracking và đếm lượt người đi qua là bài toán khác, chưa nằm trong chức năng hiện tại.
- Báo cáo phiên hiện lưu mẫu số người nhìn thấy; không định danh cá nhân hoặc đếm lượt vào/ra.
