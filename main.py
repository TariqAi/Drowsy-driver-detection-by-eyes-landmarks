from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame

COUNTER = 0
ALARM_ON = False
model_path = "model/shape_predictor_68_face_landmarks.dat"
ALARM_PATH = "alarm/alert.mp3"

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect():
    thresh = 0.25
    frame_check = 60  # (5 seconds * 30 fps)
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(model_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    flag = 0

    # Initialize pygame mixer outside the loop
    pygame.mixer.init()

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640, height=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    if not ALARM_ON:
                        ALARM_ON = True
                        # Play the alarm sound once
                        pygame.mixer.music.load(ALARM_PATH)
                        pygame.mixer.music.play()
                    cv2.putText(frame, "****************Drowsy!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, 'sleeping time 00:' + str(flag // 30) + ' sec', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag = 0
                ALARM_ON = False
                #pygame.mixer.music.stop()     # Stop the alarm sound if it is playing

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    detect()
