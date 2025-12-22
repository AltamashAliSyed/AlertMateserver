from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import math

app = FastAPI()

# ---------------- GLOBAL ALERT STATE ----------------
CURRENT_ALERT = "NORMAL"

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh

# ---------------- THRESHOLDS ----------------
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.45


# ---------------- UTILS ----------------
def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def compute_ear(lm, idx):
    p = [(lm[i].x, lm[i].y) for i in idx]
    return (euclidean(p[1], p[5]) + euclidean(p[2], p[4])) / (
        2 * euclidean(p[0], p[3])
    )

def mouth_ratio(lm, top_idx=13, bottom_idx=14, ref=1.0):
    t = (lm[top_idx].x, lm[top_idx].y)
    b = (lm[bottom_idx].x, lm[bottom_idx].y)
    return euclidean(t, b) / ref if ref != 0 else 0


# ---------------- ROOT CHECK ----------------
@app.get("/")
def root():
    return {
        "status": "AlertMate API is running",
        "detect_endpoint": "/detect",
        "status_endpoint": "/status"
    }


# ---------------- STATUS API (FOR MOBILE APP) ----------------
@app.get("/status")
def get_status():
    return {
        "alert": CURRENT_ALERT
    }


# ---------------- DETECTION API (ESP32 CALLS THIS) ----------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global CURRENT_ALERT

    contents = await file.read()

    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        CURRENT_ALERT = "NORMAL"
        return JSONResponse({"alert": CURRENT_ALERT})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            CURRENT_ALERT = "NORMAL"
            return JSONResponse({"alert": CURRENT_ALERT})

        lm = result.multi_face_landmarks[0].landmark

        LEFT_EYE  = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        ear = (
            compute_ear(lm, LEFT_EYE) +
            compute_ear(lm, RIGHT_EYE)
        ) / 2

        ref = euclidean(
            (lm[LEFT_EYE[0]].x, lm[LEFT_EYE[0]].y),
            (lm[LEFT_EYE[3]].x, lm[LEFT_EYE[3]].y)
        )

        mar = mouth_ratio(lm, ref=ref)

        # ---------------- DECISION ----------------
        if ear < EYE_AR_THRESH:
            CURRENT_ALERT = "EYE_CLOSED"
        elif mar > MOUTH_AR_THRESH:
            CURRENT_ALERT = "YAWNING"
        else:
            CURRENT_ALERT = "NORMAL"

        return JSONResponse({"alert": CURRENT_ALERT})
