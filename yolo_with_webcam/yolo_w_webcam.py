from ultralytics import YOLO
import cv2
import math
import cvzone

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

# run webcam
cap=cv2.VideoCapture(0)
cap.set(3,1280) # width
cap.set(4,720) # height

# for video
# cap=cv2.VideoCapture("../Videos/cars.mp4")

while cap.isOpened():
    success,img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:

            # for the bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            # vanilla opencv
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(102,241,255),3)
            # cool lines
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            # for the bounding box
            conf =math.ceil((box.conf[0]*100))/100 # for 2 decimal places
            # print(conf)
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(30,y1)))

            # for the class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(30, y1)))

    cv2.imshow("Image",img)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

