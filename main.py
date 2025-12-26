from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
import math
import time

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.45

EYE_TIME_THRESHOLD = 3.0   # seconds
YAWN_TIME_THRESHOLD = 3.0 # seconds

eye_start = None
yawn_start = None
CURRENT_ALERT = "NORMAL"

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def compute_ear(lm, idx):
    p = [(lm[i].x, lm[i].y) for i in idx]
    return (euclidean(p[1], p[5]) + euclidean(p[2], p[4])) / (
        2 * euclidean(p[0], p[3])
    )

def mouth_ratio(lm, ref):
    t = (lm[13].x, lm[13].y)
    b = (lm[14].x, lm[14].y)
    return euclidean(t, b) / ref if ref else 0

@app.get("/")
def root():
    return {"status": "AlertMate Cloud Running"}

@app.get("/status")
def status():
    return {"alert": CURRENT_ALERT}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global eye_start, yawn_start, CURRENT_ALERT

    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        CURRENT_ALERT = "NORMAL"
        return {"alert": CURRENT_ALERT}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        CURRENT_ALERT = "NORMAL"
        eye_start = None
        yawn_start = None
        return {"alert": CURRENT_ALERT}

    lm = result.multi_face_landmarks[0].landmark

    LEFT = [33,160,158,133,153,144]
    RIGHT = [362,385,387,263,373,380]

    ear = (compute_ear(lm, LEFT) + compute_ear(lm, RIGHT)) / 2
    ref = euclidean(
        (lm[LEFT[0]].x, lm[LEFT[0]].y),
        (lm[LEFT[3]].x, lm[LEFT[3]].y)
    )
    mar = mouth_ratio(lm, ref)

    now = time.time()

    if ear < EYE_AR_THRESH:
        eye_start = eye_start or now
        if now - eye_start >= EYE_TIME_THRESHOLD:
            CURRENT_ALERT = "EYE_CLOSED"
            return {"alert": CURRENT_ALERT}
    else:
        eye_start = None

    if mar > MOUTH_AR_THRESH:
        yawn_start = yawn_start or now
        if now - yawn_start >= YAWN_TIME_THRESHOLD:
            CURRENT_ALERT = "YAWNING"
            return {"alert": CURRENT_ALERT}
    else:
        yawn_start = None

    CURRENT_ALERT = "NORMAL"
    return {"alert": CURRENT_ALERT}
