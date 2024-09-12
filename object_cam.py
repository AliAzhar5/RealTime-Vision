import cv2
from ultralytics import YOLO
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('yolo-weights/yolov8n.pt')

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


start_time = time.time()  
image_count = 0           
save_interval = 5
max_images = 2

while True:

    ret, frame = cap.read()
    results = model(frame, stream=True)
    total_object = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil(box.conf[0]*100)/100
            print('Confidence:', confidence)

            cls = int(box.cls[0])
            print('Class:', cls)
            cv2.putText(frame, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            total_object += 1

    print(f"Total Objects Detected: {total_object}")
    cv2.putText(frame, f'Total Objects: {total_object}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    if time.time() - start_time >= save_interval and image_count < max_images:
        image_path = f'detected_frame_{image_count + 1}.jpg'
        cv2.imwrite(image_path, frame)
        print(f"Saved: {image_path}")

        image_count += 1
        start_time = time.time()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()