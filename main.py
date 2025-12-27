from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import requests

app = FastAPI()

# Allow ESP and apps to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== GLOBAL STATES =====
DEVICE_REGISTRY = {}  # device_id -> ESP IP
DEVICE_STATE = {}     # device_id -> alert, timestamps

# ===== CONFIG =====
ALERT_DURATION = 3      # seconds for eye close / yawn
COOLDOWN = 5            # seconds between alerts
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.45

# ===== MEDIAPIPE =====
mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True
)

# ===== HELPER FUNCTIONS =====
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def ear(lm, idx):
    p = [(lm[i].x, lm[i].y) for i in idx]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]))

def trigger_esp(device_id):
    ip = DEVICE_REGISTRY.get(device_id)
    if ip:
        try:
            requests.post(f"http://{ip}/alert", timeout=1)
        except:
            pass

# ===== ENDPOINTS =====
@app.get("/")
def root():
    return {"status": "AlertMate Cloud Running"}

@app.get("/status")
def status(device_id: str):
    state = DEVICE_STATE.get(device_id)
    if state:
        return {"alert": state["alert"]}
    return {"alert": "NORMAL"}

@app.post("/register_device")
async def register_device(data: dict):
    device_id = data["device_id"]
    esp_ip = data["esp_ip"]
    DEVICE_REGISTRY[device_id] = esp_ip
    DEVICE_STATE[device_id] = {"alert":"NORMAL","eye":None,"yawn":None,"last":0}
    return {"status":"registered"}

@app.post("/detect")
async def detect(file: UploadFile = File(...), device_id: str = ""):
    if device_id not in DEVICE_STATE:
        return {"alert":"NORMAL"}

    state = DEVICE_STATE[device_id]
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        state["alert"] = "NORMAL"
        return {"alert": "NORMAL"}

    res = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        state["eye"] = state["yawn"] = None
        state["alert"] = "NORMAL"
        return {"alert": "NORMAL"}

    lm = res.multi_face_landmarks[0].landmark
    LEFT = [33,160,158,133,153,144]
    RIGHT = [362,385,387,263,373,380]

    e = (ear(lm, LEFT) + ear(lm, RIGHT)) / 2
    ref = dist((lm[33].x,lm[33].y), (lm[133].x,lm[133].y))
    mar = dist((lm[13].x,lm[13].y), (lm[14].x,lm[14].y)) / ref

    now = time.time()

    # Eye close detection
    if e < EYE_AR_THRESH:
        state["eye"] = state["eye"] or now
        if now - state["eye"] >= ALERT_DURATION and now - state["last"] > COOLDOWN:
            state["alert"] = "EYE_CLOSED"
            state["last"] = now
            trigger_esp(device_id)
    else:
        state["eye"] = None

    # Yawn detection
    if mar > MOUTH_AR_THRESH:
        state["yawn"] = state["yawn"] or now
        if now - state["yawn"] >= ALERT_DURATION and now - state["last"] > COOLDOWN:
            state["alert"] = "YAWNING"
            state["last"] = now
            trigger_esp(device_id)
    else:
        state["yawn"] = None

    if state["alert"] not in ["EYE_CLOSED", "YAWNING"]:
        state["alert"] = "NORMAL"

    return {"alert": state["alert"]}
