# import cv2
# import numpy as np
# from keras.models import load_model

# model=load_model('model_file_30epochs.h5')

# video=cv2.VideoCapture(0)

# faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# labels_dict={0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Srprise'}

# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=faceDetect.detectMultiScale(gray, 1.3, 3)
#     for x,y,w,h in faces:
#         sub_face_img=gray[y:y+h, x:x+w]
#         resized=cv2.resize(sub_face_img, (48,48))
#         normalize=resized/255.0
#         reshaped=np.reshape(normalize, (1,48,48,1))
#         result=model.predict(reshaped)
#         label=np.argmax(result, axis=1)[0]
#         print(label)
#         print(labels_dict[label])
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 2)
#         cv2.rectangle(frame, (x,y-40), (x+w, y), (50,50,255), -1)
#         cv2.putText(frame, labels_dict[label], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),2)

#     cv2.imshow("Frame",frame)
#     k=cv2.waitKey(1)
#     if k==ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from keras.models import load_model
# import time

# model = load_model('model_file_30epochs.h5')

# video = cv2.VideoCapture(0)

# faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# emotion_log = []

# start_time = time.time()
# duration = 10  # Set the duration in seconds

# while (time.time() - start_time) < duration:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
#     for x, y, w, h in faces:
#         sub_face_img = gray[y:y + h, x:x + w]
#         resized = cv2.resize(sub_face_img, (48, 48))
#         normalize = resized / 255.0
#         reshaped = np.reshape(normalize, (1, 48, 48, 1))
#         result = model.predict(reshaped)
#         label = np.argmax(result, axis=1)[0]
        
#         timestamp = time.time()
#         emotion_info = {'timestamp': timestamp, 'emotion': labels_dict[label]}
#         emotion_log.append(emotion_info)

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
#         cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
#         cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()

# # Save the emotion log to a file or perform further processing
# print("Emotion log:", emotion_log)


import cv2
import numpy as np
from keras.models import load_model
import time
#import requests
from collections import Counter  # Import the Counter module

# Load the pre-trained emotion recognition model
model = load_model('model_file_30epochs.h5')

#connect to the camera
video = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

#initialize empty list to store emotion data
emotion_log = []

start_time = time.time()
duration = 5  # Set the duration in seconds

#window size changeing
def on_window_state_changed(event, x, y, flags, param):
    if event == cv2.EVENT_WINDOW_STATE_CHANGED:
        frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("Window size changed to:", frame_width, "x", frame_height)

cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Emotion Detection", on_window_state_changed)


while (time.time() - start_time) < duration:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    #process each detected face
    for x, y, w, h in faces:
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        timestamp = time.time()
        emotion_info = {'timestamp': timestamp, 'emotion': labels_dict[label]}
        emotion_log.append(emotion_info)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Emotion Detection", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
# Close all OpenCV windows
cv2.destroyAllWindows()

# Count the occurrences of each emotion in the log
emotion_counts = Counter(entry['emotion'] for entry in emotion_log)

# Get the most common emotion
most_common_emotion = emotion_counts.most_common(1)

print("Emotion log:", emotion_log)
print("Most common emotion:", most_common_emotion)
