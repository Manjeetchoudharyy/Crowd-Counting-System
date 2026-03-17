import cv2
import json
import os
import time
import csv
from datetime import datetime
from person_detector import detect_people
from tracker import Tracker

tracker = Tracker()

zones = []
drawing = False
start_point = None
zone_file = "zones.json"

fullscreen = False

zone_entry = {}
zone_exit = {}
inside_ids = {}

MAX_PEOPLE = 2

csv_file = "crowd_data.csv"

if not os.path.exists(csv_file):
    with open(csv_file,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time","Zone","Inside","Entry","Exit"])

def log_data(zone,inside,entry,exit):
    with open(csv_file,"a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            f"Zone {zone+1}",
            inside,
            entry,
            exit
        ])

def save_zones():
    with open(zone_file,"w") as f:
        json.dump(zones,f)

def load_zones():
    global zones
    if os.path.exists(zone_file):
        with open(zone_file,"r") as f:
            zones = json.load(f)

def mouse_draw(event,x,y,flags,param):

    global drawing,start_point,zones

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x,y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x,y)
        zones.append([start_point,end_point])

load_zones()

cap = cv2.VideoCapture(0)

cv2.namedWindow("Crowd Camera")
cv2.setMouseCallback("Crowd Camera",mouse_draw)

colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255)]

while True:

    ret,frame = cap.read()
    if not ret:
        break

    detections = detect_people(frame)
    boxes_ids = tracker.update(detections)

    for i in range(len(zones)):
        if i not in zone_entry:
            zone_entry[i] = 0
            zone_exit[i] = 0
            inside_ids[i] = set()

    for box in boxes_ids:

        x1,y1,x2,y2,id = box

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

        cv2.putText(frame,f"ID:{id}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)

        for i,zone in enumerate(zones):

            pt1 = tuple(zone[0])
            pt2 = tuple(zone[1])

            if pt1[0] < cx < pt2[0] and pt1[1] < cy < pt2[1]:

                if id not in inside_ids[i]:
                    inside_ids[i].add(id)
                    zone_entry[i] += 1

            else:

                if id in inside_ids[i]:
                    inside_ids[i].remove(id)
                    zone_exit[i] += 1

    for i,zone in enumerate(zones):

        pt1 = tuple(zone[0])
        pt2 = tuple(zone[1])

        color = colors[i % len(colors)]

        cv2.rectangle(frame,pt1,pt2,color,2)

        inside = len(inside_ids[i])

        cv2.putText(frame,f"Zone {i+1}",
                    (pt1[0],pt1[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1)

        cv2.putText(frame,f"Inside: {inside}",
                    (pt1[0],pt1[1]+15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,255,255),
                    1)

        cv2.putText(frame,f"Entry: {zone_entry[i]}",
                    (pt1[0],pt1[1]+30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255,255,0),
                    1)

        cv2.putText(frame,f"Exit: {zone_exit[i]}",
                    (pt1[0],pt1[1]+45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,255,255),
                    1)

        log_data(i,inside,zone_entry[i],zone_exit[i])

        if inside > MAX_PEOPLE:

            cv2.putText(frame,
                        "OVER CROWD ALERT!",
                        (200,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

            if not os.path.exists("alerts"):
                os.makedirs("alerts")

            filename = f"alerts/alert_{int(time.time())}.jpg"
            cv2.imwrite(filename,frame)

    cv2.putText(frame,
                "Draw Zone: Mouse Drag",
                (10,20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255,255,255),
                1)

    cv2.putText(frame,
                "S:Save D:Delete R:Reset P:Screenshot F:Fullscreen ESC:Exit",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255,255,255),
                1)

    cv2.imshow("Crowd Camera",frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        save_zones()

    elif key == ord('d'):
        if zones:
            zones.pop()

    elif key == ord('r'):
        zones.clear()

    elif key == ord('p'):
        cv2.imwrite("zone_screenshot.png",frame)

    elif key == ord('f'):

        fullscreen = not fullscreen

        if fullscreen:
            cv2.setWindowProperty("Crowd Camera",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Crowd Camera",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL)

    elif key == 27:
        save_zones()
        break

cap.release()
cv2.destroyAllWindows()