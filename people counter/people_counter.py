import numpy as np
from ultralytics import YOLO
import cv2
import math
import cvzone
from sort import *

# creating model
model=YOLO("../yolo_weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask=cv2.imread("mask.png")
# tracker instance
tracker=Sort(max_age=20,min_hits=4,iou_threshold=0.7)

limits_up=[103,161,296,161] # customized acc to the cam position for the lane
limits_down=[527,489,735,489] # customized acc to the cam position for the lane

# run webcam
#
# cap=cv2.VideoCapture(1)
# cap.set(3,1280) # width
# cap.set(4,720) # height

# for video
cap=cv2.VideoCapture("../Videos/people.mp4")
total_count_up=[]
total_count_down=[]
while cap.isOpened():
    success,img=cap.read()
    # with mask
    img_region=cv2.bitwise_and(img,mask)

    results=model(img_region,stream=True)

    # for the Sort dep format
    detections=np.empty((0,5))

    for r in results:
        boxes=r.boxes
        for box in boxes:

            # for the bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            
            w,h=x2-x1,y2-y1

            # for the bounding box
            conf =math.ceil((box.conf[0]*100))/100 # for 2 decimal places
            
            # for the class name
            cls = int(box.cls[0])
            current_class=classNames[cls]

            if current_class=="person" and conf>0.3:
                

                current_array=np.array([x1,y1,x2,y2,conf])
                # a vertical stack
                detections=np.vstack((detections,current_array))

    tracker_results=tracker.update(detections)
    cv2.line(img,(limits_up[0],limits_up[1]),(limits_up[2],limits_up[3]),(0,0,255),5)
    cv2.line(img,(limits_down[0],limits_down[1]),(limits_down[2],limits_down[3]),(0,0,255),5)

    for result in tracker_results:

        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=7,rt=2,colorR=(0,0,255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(30, y1)), scale=1, thickness=1, offset=7)


    # finding the center
    cx,cy=x1+w//2,y1+h//2
    cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)

    if limits_up[0]<cx<limits_up[2] and limits_up[1]-20<cy<limits_up[1]+20:
        if total_count_up.count(id)==0:
            total_count_up.append(id)
            cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 5)
    #
    cvzone.putTextRect(img, f'count up= {len(total_count_up)}',(700,50))
    cvzone.putTextRect(img, f'count down= {len(total_count_down)}',(700,100))

    cv2.imshow("Image",img)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

