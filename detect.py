import cv2
import os
import glob
import time
from ultralytics import YOLO

# Load YOLOv8 model (use yolov8n.pt for Jetson Nano)
model = YOLO("yolov8n.pt")

# ------------------------------
# IMAGE DETECTION FUNCTION
# ------------------------------
def detect_on_images():
    os.makedirs("output/detect/predict", exist_ok=True)
    image_paths = glob.glob("input/*.jpg") + glob.glob("input/*.png")
    for idx, img_path in enumerate(image_paths, start=1):
        results = model(img_path, save=True, save_txt=False, project="output", name="detect/predict")
        print(f" Processed {img_path} -> Saved to output/detect/predict/image_{idx}.jpg")


# ------------------------------
# VIDEO FILE DETECTION FUNCTION
# ------------------------------
def detect_on_videos():
    video_paths = glob.glob("input/*.mp4") + glob.glob("input/*.avi")
    for idx, video_path in enumerate(video_paths, start=1):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

        out_path = f"output/detect/predict/video_{idx}.avi"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f" Processed {video_path} -> Saved to {out_path}")


# ------------------------------
# REAL-TIME WEBCAM DETECTION FUNCTION (6 SECONDS)
# ------------------------------
def detect_from_webcam_fixed_duration(duration=6):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/detect/predict/webcam_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Failed to open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    video_path = os.path.join(output_dir, "output.avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

    start_time = time.time()
    img_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time > duration:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        frame_path = os.path.join(output_dir, f"frame_{img_count}.jpg")
        cv2.imwrite(frame_path, annotated_frame)
        out.write(annotated_frame)

        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        img_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f" Webcam detection complete. Frames and video saved to {output_dir}")


# ------------------------------
# MAIN FUNCTION – CHOOSE WHAT TO RUN
# ------------------------------
def main():
    print(" Starting YOLOv8 Detection Tasks...")

    detect_on_images()
    detect_on_videos()
    detect_from_webcam_fixed_duration(duration=6)

    print(" All tasks completed.")


if __name__ == "__main__":
    main()