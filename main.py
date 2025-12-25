from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp
import math
import time

app = FastAPI()

# ===================== GLOBAL STATE =====================
CURRENT_ALERT = "NORMAL"

eye_closed_start = None
yawn_start = None

ALERT_TIME = 3.0  # seconds

# ===================== MEDIAPIPE =====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ===================== THRESHOLDS =====================
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.45

# ===================== HELPERS =====================
def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def compute_ear(lm, idx):
    p = [(lm[i].x, lm[i].y) for i in idx]
    return (euclidean(p[1], p[5]) + euclidean(p[2], p[4])) / (
        2 * euclidean(p[0], p[3])
    )

def mouth_ratio(lm, ref):
    t = (lm[13].x, lm[13].y)
    b = (lm[14].x, lm[14].y)
    return euclidean(t, b) / ref if ref else 0

# ===================== ROUTES =====================
@app.get("/")
def root():
    return {"status": "AlertMate API running"}

@app.get("/status")
def status():
    return {"alert": CURRENT_ALERT}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global CURRENT_ALERT
    global eye_closed_start, yawn_start

    contents = await file.read()
    img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if frame is None:
        CURRENT_ALERT = "NORMAL"
        eye_closed_start = None
        yawn_start = None
        return {"alert": CURRENT_ALERT}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        CURRENT_ALERT = "NORMAL"
        eye_closed_start = None
        yawn_start = None
        return {"alert": CURRENT_ALERT}

    lm = result.multi_face_landmarks[0].landmark

    LEFT = [33, 160, 158, 133, 153, 144]
    RIGHT = [362, 385, 387, 263, 373, 380]

    ear = (compute_ear(lm, LEFT) + compute_ear(lm, RIGHT)) / 2

    ref = euclidean(
        (lm[LEFT[0]].x, lm[LEFT[0]].y),
        (lm[LEFT[3]].x, lm[LEFT[3]].y)
    )
    mar = mouth_ratio(lm, ref)

    now = time.time()

    # ===================== EYE CLOSED LOGIC =====================
    if ear < EYE_AR_THRESH:
        if eye_closed_start is None:
            eye_closed_start = now
        elif now - eye_closed_start >= ALERT_TIME:
            CURRENT_ALERT = "EYE_CLOSED"
            return {"alert": CURRENT_ALERT}
    else:
        eye_closed_start = None

    # ===================== YAWNING LOGIC =====================
    if mar > MOUTH_AR_THRESH:
        if yawn_start is None:
            yawn_start = now
        elif now - yawn_start >= ALERT_TIME:
            CURRENT_ALERT = "YAWNING"
            return {"alert": CURRENT_ALERT}
    else:
        yawn_start = None

    # ===================== NORMAL =====================
    CURRENT_ALERT = "NORMAL"
    return {"alert": CURRENT_ALERT}
