from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import requests

app = FastAPI()

# ================= GLOBAL STATE =================
CURRENT_ALERT = "NORMAL"

DEVICE_REGISTRY = {}  # device_id -> esp_ip

eye_start = None
yawn_start = None

ALERT_DURATION = 3.0  # seconds
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.45

# ================= MEDIAPIPE (CPU SAFE) =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ================= UTILS =================
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

def trigger_esp(device_id):
    esp_ip = DEVICE_REGISTRY.get(device_id)
    if not esp_ip:
        return
    try:
        requests.get(f"http://{esp_ip}/alert", timeout=1)
    except:
        pass

# ================= ROUTES =================
@app.get("/")
def root():
    return {"status": "AlertMate Cloud Running"}

@app.get("/status")
def status():
    return {"alert": CURRENT_ALERT}

@app.post("/register_device")
async def register_device(data: dict):
    """
    Body:
    {
      "device_id": "driver123",
      "esp_ip": "192.168.4.1"
    }
    """
    DEVICE_REGISTRY[data["device_id"]] = data["esp_ip"]
    return {"status": "registered"}

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    device_id: str = "driver123"
):
    global CURRENT_ALERT, eye_start, yawn_start

    contents = await file.read()
    img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if frame is None:
        CURRENT_ALERT = "NORMAL"
        return {"alert": CURRENT_ALERT}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        eye_start = None
        yawn_start = None
        CURRENT_ALERT = "NORMAL"
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

    # ===== EYE CLOSED =====
    if ear < EYE_AR_THRESH:
        if eye_start is None:
            eye_start = now
        elif now - eye_start >= ALERT_DURATION:
            CURRENT_ALERT = "EYE_CLOSED"
            trigger_esp(device_id)
            return {"alert": CURRENT_ALERT}
    else:
        eye_start = None

    # ===== YAWNING =====
    if mar > MOUTH_AR_THRESH:
        if yawn_start is None:
            yawn_start = now
        elif now - yawn_start >= ALERT_DURATION:
            CURRENT_ALERT = "YAWNING"
            trigger_esp(device_id)
            return {"alert": CURRENT_ALERT}
    else:
        yawn_start = None

    CURRENT_ALERT = "NORMAL"
    return {"alert": CURRENT_ALERT}
