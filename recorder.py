import cv2
import csv
import keyboard
import mediapipe as mp


model_path = "C:\\Users\\benja\\operation-posture\\pose_landmarker_full.task"
csv_file = "C:\\Users\\benja\\operation-posture\\data.csv"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode





print("Waiting to record...")

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # g --> good, l --> lean, h --> hunch
    if keyboard.is_pressed("g") or keyboard.is_pressed("l") or keyboard.is_pressed("h"):
        if result.pose_landmarks:
            flattened_results = []
            # first person seen
            for landmark in result.pose_landmarks[0]:
                flattened_results.append(landmark.x)
                flattened_results.append(landmark.y)
                flattened_results.append(landmark.z)
                flattened_results.append(landmark.visibility)
            # subject to change
            if keyboard.is_pressed("g"):
                flattened_results.append("Good")
            elif keyboard.is_pressed("l"):
                flattened_results.append("Lean")
            elif keyboard.is_pressed("h"):
                flattened_results.append("Hunch")
            writer.writerow(flattened_results)
        else:
            print("No human detected, unable to save landmarks")
    else:
        print("KEY IS NOT PRESSED")


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

    
    

with PoseLandmarker.create_from_options(options) as landmarker:

    cam = cv2.VideoCapture(0)
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        
        frame_timestamp_ms = 0
        while cam.isOpened():

            if keyboard.is_pressed("q"):
                print("Recording stopped")
                break

            ret, frame = cam.read()
            # frame isnt processed
            if not ret:
                break
            # frames come out in bgr, which throws off pose detection trained with rgb
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # not sure if this needs a for loop
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            cv2.imshow('Recorder', frame)
            cv2.waitKey(1)

            # frames need to be in ascending order, this should be fine.
            frame_timestamp_ms += 1
            
            # sends the image data to perform the landmarking. second arg is the timestamp of the frame
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            # print("Running...")

            