import cv2
import mediapipe as mp

# import keyboard
import time

cap = cv2.VideoCapture(0)
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

    running = True

    while running:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        frame_timestamp_ms = int(time.time() * 1000)
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        recognition_result = recognizer.recognize_async(mp_img, frame_timestamp_ms)
        
        try:
            cv2.putText(img, f"Gesture: {cname}", (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)
        except:
            pass

        cv2.imshow("Gesture Recognition", img)
        
        # if keyboard.is_pressed(' '):
        #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()