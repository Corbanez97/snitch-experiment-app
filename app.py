import cv2
import streamlit as st

import mediapipe as mp

import time

st.title("Snitch :point_up: ")
run = st.button('Avaliar Higienização')
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
init_frame = 0
capture = False

if not cap.isOpened():
    print("Error: Could not open video file")
    
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def add_result_to_img(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global cname
    try:
        cname = result.gestures[0][0].category_name
    except:
        cname = 'None'

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='models/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=add_result_to_img
    )

with GestureRecognizer.create_from_options(options) as recognizer:

    while run:

        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        frame_timestamp_ms = int(time.time() * 1000)
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        recognition_result = recognizer.recognize_async(mp_img, frame_timestamp_ms)
        
        try:
            cv2.putText(imgRGB, f"Gesture: {cname}", (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)

            if cname == 'Open_Palm' and init_frame == 0:
                init_frame = frame_timestamp_ms
            elif cname == 'Open_Palm' and (frame_timestamp_ms - init_frame) > 3000:
                cv2.imwrite("captures/to_evaluate.jpg", img)
                capture = True
                break
            elif cname != 'Open_Palm':
                init_frame = 0
        except:
            pass

        FRAME_WINDOW.image(imgRGB)
    
if capture == True:
    st.success('Imagem capturada')