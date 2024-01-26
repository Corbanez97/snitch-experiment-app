import streamlit as st
import pandas as pd

import cv2
import mediapipe as mp

import time
import datetime

import uuid

import json

global CAMERA
CAMERA = 0

def on_page_load():
    st.set_page_config(layout="wide")

courses = pd.read_csv('data/courses.csv')

# run first
on_page_load()   

st.title("Snitch :point_up: ")

st.markdown("---")

col1, col2 = st.columns([1,1])

with col1:

    with st.form("Registro"):

        myuuid = uuid.uuid4()

        name = st.text_input('Nome', '')

        bday = st.date_input(
            'Data de nascimento',
            min_value = datetime.date(1960, 1, 1)
            )

        degree = st.selectbox(
            'Formação acadêmica', 
            [
                'Médio em andamento',
                'Médio completo',
                'Superior em andamento',
                'Superior completo',
                'Mestrado em andamento',
                'Mestrado completo',
                'Doutorado em andamento',
                'Doutorado completo'
                ]
            )

        area = st.selectbox('Curso', courses['courses'].unique())

        exp = st.selectbox(
            'Experiência em ambientes hospitalares?', 
            [
                'Sim', 
                'Não'
                ]
            )

        run = st.form_submit_button('Validar Higienização')

        if run:
            data = dict(
                {
                    'name': name,
                    'birth_date': bday.strftime("%Y-%m-%d"),
                    'degree': degree,
                    'area': area,
                    'health_experience': exp,
                    'capture_id': str(myuuid)
                }
            )

with col2:

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

    if run:
        st.subheader('Mantenha sua mão aberta')

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(CAMERA)

    if not cap.isOpened():
        print("Error: Could not open video file")

    init_frame = 0
    capture = False

    while run:

        with GestureRecognizer.create_from_options(options) as recognizer:

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
                    cv2.imwrite(f"captures/{str(myuuid)}.jpg", img)
                    capture = True
                    break
                    
                elif cname != 'Open_Palm':
                    init_frame = 0

            except:
                pass

            FRAME_WINDOW.image(imgRGB)

st.markdown('---')

if capture:
    with open(f'forms/{str(myuuid)}.json', 'w') as fp:
        json.dump(data, fp)
    st.success('Validação finalizada')