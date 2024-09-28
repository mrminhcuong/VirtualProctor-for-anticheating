import cv2
from cheatdetection import CheatDetection
from utils import config
import os
def main():
    cwd = os.getcwd()
    while True:
        with_angle = input("Need angle? (true/false): ").strip().lower()
        if with_angle == "true":
            model_name = 'weights.angle.keras'
            model_path = os.path.join(cwd, model_name)
            with_angle_bool = True
            break
        elif with_angle == "false":
            model_name = 'weights.best.keras'
            model_path = os.path.join(cwd, model_name)
            with_angle_bool = False
            break
        else:
            print("Invalid input. Please enter 'true' or 'false'.")            
    cheat_detector = CheatDetection(config, model_path, with_angle_bool)
    frame_skip_constant=3
    frame_counter=0
    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 for default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_counter % frame_skip_constant != 0:
            frame_counter += 1
            continue

        # Detect cheating in the current frame
        output_frame, cheating_detected = cheat_detector.detect_cheating(frame)

        # Display the frame
        cv2.imshow('Cheat Detection', output_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1
    

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
