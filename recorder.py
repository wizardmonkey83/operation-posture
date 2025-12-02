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


with open(csv_file, "w") as file:
    writer = csv.writer(file)


print("Waiting to record...")
recording = False
if keyboard.is_pressed("r"):
    recording = True
    print("Recording in progress...")

if keyboard.is_pressed("q"):
    recording = False
    print("Recording stopped")

cam = cv2.VideoCapture(0)
while cam.isOpened():

    def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if keyboard.is_pressed("q"):
            if result.pose_landmarks:
                flattened_results = []
                # first person seen
                for landmark in result.pose_landmarks[0]:
                    flattened_results.append(landmark.x)
                    flattened_results.append(landmark.y)
                    flattened_results.append(landmark.z)
                    flattened_results.append(landmark.visibility)
                # subject to change
                flattened_results.append("Good")
                writer.writerow(flattened_results)
            else:
                print("No human detected, unable to save landmarks")
        else:
            print("Recording is not in progress, unable to save landmarks")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)

    with PoseLandmarker.create_from_options(options) as landmarker:

        frame_timestamp_ms = 0
        ret, frame = cam.read()
        # frame isnt processed
        if not ret:
            break
        # frames come out in bgr, which throws off pose detection trained with rgb
        rgb_frame = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB)
        # not sure if this needs a for loop
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # frames need to be in ascending order, this should be fine.
        frame_timestamp_ms += 1
        
        # sends the image data to perform the landmarking. second arg is the timestamp of the frame
        landmarker.detect_async(mp_image, frame_timestamp_ms)