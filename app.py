import cv2
import time
import mediapipe as mp
import simpleaudio as sa
import threading


# CONFIG APP
EYE_CLOSED_THRESHOLD = 0.20
CLOSED_TIME_LIMIT = 5
ALARM_WAV = "assets/sound/alert-dawg.wav"


# MEDIAPIPE SETUP
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
    vert1 = ((p2.x - p6.x)**2 + (p2.y - p6.y)**2) ** 0.5
    vert2 = ((p3.x - p5.x)**2 + (p3.y - p5.y)**2) ** 0.5
    horiz = ((p1.x - p4.x)**2 + (p1.y - p4.y)**2) ** 0.5
    return (vert1 + vert2) / (2.0 * horiz)


# ALARM CONTROL
alarm_wave = sa.WaveObject.from_wave_file(ALARM_WAV)
play_obj = None
alarm_active = False

def play_alarm_loop():
    global play_obj, alarm_active
    while alarm_active:
        play_obj = alarm_wave.play()
        play_obj.wait_done()

def stop_alarm():
    global play_obj
    if play_obj:
        play_obj.stop()


# CAMERA
cap = cv2.VideoCapture(0)
eye_closed_start = None

print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        ear = (
            eye_aspect_ratio(landmarks, LEFT_EYE) +
            eye_aspect_ratio(landmarks, RIGHT_EYE)
        ) / 2

        if ear < EYE_CLOSED_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            else:
                elapsed = time.time() - eye_closed_start
                cv2.putText(frame, f"Atmin tidur cik :D {int(elapsed)} dtk",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

                if elapsed >= CLOSED_TIME_LIMIT and not alarm_active:
                    alarm_active = True
                    threading.Thread(
                        target=play_alarm_loop,
                        daemon=True
                    ).start()
        else:
            eye_closed_start = None
            if alarm_active:
                alarm_active = False
                stop_alarm()

    cv2.imshow("Eye Alarm", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()