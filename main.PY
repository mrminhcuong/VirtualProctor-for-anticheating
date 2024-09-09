import cv2
from cheatdetection import CheatDetection
from utils import config
def main():
    model_path = "/home/huyn/anticheating/weights.angle.keras"  # Update with your model path
    cheat_detector = CheatDetection(config, model_path)
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
