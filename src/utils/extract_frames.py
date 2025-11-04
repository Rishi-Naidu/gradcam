import cv2

def extract_frames(video_path):
    """
    Reads a video and returns all frames as a list of numpy arrays.
    Does NOT write anything to disk.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames
