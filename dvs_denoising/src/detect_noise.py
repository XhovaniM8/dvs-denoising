import cv2
import numpy as np

def detect_noise(video_path, diff_threshold=40, var_threshold=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return []

    prev_gray = None
    noisy_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = np.mean(diff)
            variance = np.var(diff)

            if mean_diff > diff_threshold or variance > var_threshold:
                noisy_frames.append((frame_idx, mean_diff, variance))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    print(f"\n📽 {video_path}")
    print(f"Total frames: {frame_idx}")
    print(f"Noisy frames detected: {len(noisy_frames)}")
    for idx, mdiff, var in noisy_frames[:10]:
        print(f"  Frame {idx:4d} | Mean Δ: {mdiff:6.2f} | Var: {var:7.2f}")

    return noisy_frames


if __name__ == "__main__":
    videos = [
        "../data/raw_video.mp4",
        "../data/filtered_video.mp4",
        "../data/output.mp4"
    ]

    for path in videos:
        detect_noise(path)
