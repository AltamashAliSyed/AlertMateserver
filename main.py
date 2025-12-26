from fastapi import FastAPI, UploadFile, File
import cv2, math, time
import numpy as np
import mediapipe as mp

app = FastAPI()

CURRENT_ALERT = "NORMAL"

# ---------- MediaPipe ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.45
ALERT_TIME = 3.0  # seconds

eye_start = None
yawn_start = None

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def eye_ratio(lm, idx):
    p = [(lm[i].x, lm[i].y) for i in idx]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]))

@app.get("/")
def root():
    return {"status": "AlertMate running"}

@app.get("/status")
def status():
    return {"alert": CURRENT_ALERT}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global CURRENT_ALERT, eye_start, yawn_start

    img = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if frame is None:
        CURRENT_ALERT = "NORMAL"
        return {"alert": CURRENT_ALERT}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        CURRENT_ALERT = "NORMAL"
        eye_start = yawn_start = None
        return {"alert": CURRENT_ALERT}

    lm = result.multi_face_landmarks[0].landmark

    LEFT = [33,160,158,133,153,144]
    RIGHT = [362,385,387,263,373,380]

    ear = (eye_ratio(lm, LEFT) + eye_ratio(lm, RIGHT)) / 2

    mouth_open = dist(
        (lm[13].x, lm[13].y),
        (lm[14].x, lm[14].y)
    )

    now = time.time()

    # ----- Eye closed -----
    if ear < EYE_AR_THRESH:
        if eye_start is None:
            eye_start = now
        elif now - eye_start >= ALERT_TIME:
            CURRENT_ALERT = "EYE_CLOSED"
            return {"alert": CURRENT_ALERT}
    else:
        eye_start = None

    # ----- Yawning -----
    if mouth_open > MOUTH_AR_THRESH:
        if yawn_start is None:
            yawn_start = now
        elif now - yawn_start >= ALERT_TIME:
            CURRENT_ALERT = "YAWNING"
            return {"alert": CURRENT_ALERT}
    else:
        yawn_start = None

    CURRENT_ALERT = "NORMAL"
    return {"alert": CURRENT_ALERT}
