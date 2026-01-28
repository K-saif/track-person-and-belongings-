
# ================================
#  Helper: Detection + Track Split
# ================================

def run_yolo(model, frame):
    results = model(frame, conf=0.4, iou=0.4, classes=[0, 24, 26, 28], verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf, cls = float(box.conf[0]), int(box.cls[0])
            if (x2-x1) >= 40 and (y2-y1) >= 80:
                detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
    return sorted(detections, key=lambda x: x[1], reverse=True)


def split_tracks(tracks):
    person_tracks, bag_tracks = {}, {}
    for trk in tracks:
        if trk.is_confirmed():
            tid, cls = trk.track_id, trk.get_det_class()
            l, t, r, b = trk.to_ltrb()
            if cls == 0:
                person_tracks[tid] = (l, t, r, b)
            elif cls in [24, 26, 28]:
                bag_tracks[tid] = (l, t, r, b)
    return person_tracks, bag_tracks
