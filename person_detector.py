from ultralytics import YOLO
import math

model = YOLO("yolov8n.pt")

def is_close(box1, box2):
    x1,y1,x2,y2 = box1
    a1,b1,a2,b2 = box2

    cx1 = (x1+x2)//2
    cy1 = (y1+y2)//2

    cx2 = (a1+a2)//2
    cy2 = (b1+b2)//2

    distance = math.hypot(cx1-cx2, cy1-cy2)

    return distance < 50  # threshold

def detect_people(frame):

    results = model(frame, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # person
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append((x1,y1,x2,y2))

    # REMOVE DUPLICATES
    filtered = []

    for box in detections:
        duplicate = False

        for f in filtered:
            if is_close(box, f):
                duplicate = True
                break

        if not duplicate:
            filtered.append(box)

    return filtered