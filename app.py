from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import math
from collections import defaultdict
import time
from tracker import PersonBagTracker
from utils import run_yolo, split_tracks

# ================================
#  Main Pipeline
# ================================

def main():
    start_time = time.time()
    model = YOLO("yolo11n.pt")
    tracker = DeepSort(max_age=30, n_init=5, embedder="mobilenet", half=True)

    cap = cv2.VideoCapture("1.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter("output2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    logic = PersonBagTracker()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        detections = run_yolo(model, frame)
        tracks = tracker.update_tracks(detections, frame=frame)

        person_tracks, bag_tracks = split_tracks(tracks)

        logic.associate(person_tracks, bag_tracks)
        logic.confirm_ownership()
        logic.check_bag_release(person_tracks, bag_tracks)
        logic.check_separation(person_tracks, bag_tracks, frame)
        logic.check_person_exit(person_tracks, frame)

        logic.draw_boxes(frame, tracks)
        logic.draw_left_behind(frame, bag_tracks)

        out.write(frame)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✔ DONE — Saved output2.mp4 in {time.time()-start_time:.2f}s")


# ================================
#  Run
# ================================

if __name__ == "__main__":
    main()
