import cv2
import numpy as np

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2

# Initialize video capture
cap = cv2.VideoCapture(1)  # Change the index to 0 if it's the first camera
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
cap.set(10, 150)  # Set brightness

classnames = []
classfile = r"C:\Users\91738\Desktop\VS code all files\Open cv project\coco.names"

# Read class names from file
with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

print(classnames)

configpath = r"C:\Users\91738\Desktop\VS code all files\Open cv project\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightspath = r"C:\Users\91738\Desktop\VS code all files\Open cv project\frozen_inference_graph.pb"

# Load the pre-trained model from OpenCV documentation
net = cv2.dnn_DetectionModel(weightspath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        break

    # Detect objects
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Draw bounding box and label
            cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)
            cv2.putText(img, classnames[classId-1].upper(), (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) #display the bounding box in red colour and text in blue colour

    # Display the resulting frame
    cv2.imshow('Output', img)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
