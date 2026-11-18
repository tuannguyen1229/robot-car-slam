# robot-car-slam

Xe Ackermann chạy trên Raspberry Pi. Dùng RPLiDAR A1 với Cartographer để vẽ bản đồ, sau đó click một điểm trên map trong Foxglove là xe tự chạy tới đó (goal ở phía sau thì xe tự lùi). Ngoài ra có thêm trang web để điều khiển tay và nhận diện biển báo bằng camera.

Pi nói chuyện với Arduino Mega 2560 qua cổng USB, 115200 baud. Mega lo servo lái, motor và encoder.

## Có gì trong repo

- `slam/`: chạy LiDAR, TF, Cartographer và Foxglove Bridge
- `navigation/`: `goal_pose_drive_reverse_ppnew11.py` là bộ điều khiển chính (pure pursuit, lùi, quay đầu, né vật cản), `goto_clicked_point.py` đổi điểm click thành goal
- `ai_control/`: web điều khiển + nhận diện biển báo bằng Edge Impulse
- `firmware/`: code Arduino Mega, mỗi sketch một thư mục

## Chạy

ROS 2 Jazzy, mỗi dòng một terminal:

```
./slam/start_complete_slam_system.sh
python3 navigation/goal_pose_drive_reverse_ppnew11.py
python3 navigation/goto_clicked_point.py
./ai_control/start_ai_robot_control.sh
```

Foxglove kết nối tới `ws://<ip-pi>:8765`, web điều khiển ở `http://<ip-pi>:8080`.

## UART

Pi gửi cho Mega:
- `V,servo_us,pwm` tiến
- `B,servo_us,pwm` lùi
- `S` dừng

Mega gửi lại `E,trái,phải,servo_us` (số xung encoder hai bánh và xung servo đang phát).

## Số đo xe

Lấy từ `goal_pose_drive_reverse_ppnew11.py`:

- chiều dài cơ sở 20 cm, thân xe rộng 20 cm
- từ trục sau tới mũi xe 23 cm, tới đuôi 4 cm
- LiDAR đặt trước trục sau 10 cm
- bánh đường kính 65 mm, encoder 329.8 xung/vòng
- góc lái tối đa 40 độ
- servo: trái 1650, giữa 2125, phải 2600 (µs)

## Mấy chỗ cần để ý

- File model `.eim` không có trong repo. Đường dẫn model đang để cứng `/home/quan/edge-impulse-linux-runner/...` ở `ai_control/realtime_detect_with_control.py` và `ai_control/start_ai_robot_control.sh`, máy khác thì sửa lại.
- Servo giữa trong firmware `ai_car_simple_v2_goal_nav.ino` là 2300 còn bên Python là 2125, hơi lệch nhau. Nếu xe bị lệch lái lúc dừng thì kiểm tra chỗ này.
