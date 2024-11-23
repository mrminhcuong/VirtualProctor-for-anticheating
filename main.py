import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from cheatdetection import CheatDetection
from utils import config

class VirtualProctorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Proctor - Anti-Cheating System")
        self.setGeometry(100, 100, 800, 600)

        # Thiết lập giao diện
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Nhãn hiển thị video
        self.video_label = QLabel("Video Preview")
        self.video_label.setStyleSheet("background-color: #000;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        # ComboBox chọn kiểu kiểm tra
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("With Angle Detection")
        self.mode_selector.addItem("Without Angle Detection")
        self.layout.addWidget(self.mode_selector)

        # ComboBox chọn camera
        self.camera_selector = QComboBox()
        self.detect_cameras()  # Tự động phát hiện camera
        self.layout.addWidget(self.camera_selector)

        # Nút bắt đầu
        self.start_button = QPushButton("Start Monitoring")
        self.start_button.clicked.connect(self.start_monitoring)
        self.layout.addWidget(self.start_button)

        # Nút dừng
        self.stop_button = QPushButton("Stop Monitoring")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)  # Chỉ bật khi đang giám sát
        self.layout.addWidget(self.stop_button)

        # Trạng thái
        self.status_label = QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        # Cấu hình camera và cheat detection
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.camera = None
        self.cheat_detector = None
        self.frame_counter = 0
        self.frame_skip_constant = 3
        self.current_camera_id = 0  # ID của camera đang sử dụng

    def detect_cameras(self):
        """
        Tự động phát hiện camera có sẵn và thêm vào combo box.
        """
        for i in range(5):  # Thử kiểm tra tối đa 5 camera
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_selector.addItem(f"Camera {i}")
                cap.release()

    def start_monitoring(self):
        mode = self.mode_selector.currentText()
        with_angle = mode == "With Angle Detection"
        self.status_label.setText(f"Status: Monitoring ({mode})")

        # Tải mô hình dựa trên lựa chọn
        cwd = os.getcwd()
        model_name = 'weights.angle.keras' if with_angle else 'weights.best.keras'
        model_path = os.path.join(cwd, model_name)
        self.cheat_detector = CheatDetection(config, model_path, with_angle)

        # Lấy camera ID từ combo box
        self.current_camera_id = self.camera_selector.currentIndex()

        # Khởi động camera
        self.camera = cv2.VideoCapture(self.current_camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.camera.isOpened():
            self.status_label.setText(f"Error: Unable to open Camera {self.current_camera_id}")
            return

        self.timer.start(30)  # Cập nhật mỗi 30ms
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_monitoring(self):
        self.timer.stop()
        if self.camera:
            self.camera.release()
        self.video_label.clear()
        self.status_label.setText("Status: Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            self.stop_monitoring()
            return

        # Skip frames
        if self.frame_counter % self.frame_skip_constant != 0:
            self.frame_counter += 1
            return

        # Phát hiện gian lận
        output_frame, cheating_detected = self.cheat_detector.detect_cheating(frame)

        # Hiển thị kết quả trên giao diện
        frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        q_image = QImage(frame.data, width, height, channel * width, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

        self.frame_counter += 1

        # Cập nhật trạng thái
        if cheating_detected:
            self.status_label.setText("Status: Cheating Detected!")
        else:
            self.status_label.setText("Status: Monitoring...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VirtualProctorApp()
    window.show()
    sys.exit(app.exec_())

def log_violation_time(self):
        """
        Ghi nhận thời gian vi phạm vào danh sách.
        """
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        self.violation_list.addItem(f"Violation at {current_time}")

