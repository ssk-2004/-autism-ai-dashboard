import cv2
import time

def get_eye_contact_score(duration=5):

    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        return 0.5

    start = time.time()

    eye_frames = 0

    total_frames = 0

    while time.time() - start < duration:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        if len(eyes) > 0:
            eye_frames += 1

        total_frames += 1

    cap.release()

    cv2.destroyAllWindows()

    if total_frames == 0:
        return 0.5

    return eye_frames / total_frames

