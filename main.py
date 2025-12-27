from fastapi import FastAPI, UploadFile, File
import cv2, numpy as np, mediapipe as mp, math, time, requests

app = FastAPI()

DEVICE_STATE = {}
DEVICE_REGISTRY = {}

ALERT_DURATION = 3
COOLDOWN = 5
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.45

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def ear(lm, idx):
    p = [(lm[i].x, lm[i].y) for i in idx]
    return (dist(p[1],p[5]) + dist(p[2],p[4])) / (2*dist(p[0],p[3]))

def trigger_esp(device_id):
    ip = DEVICE_REGISTRY.get(device_id)
    if ip:
        try:
            requests.post(f"http://{ip}/alert", timeout=1)
        except:
            pass

@app.get("/")
def root():
    return {"status":"AlertMate Cloud Running"}

@app.get("/status")
def status(device_id: str):
    return {"alert": DEVICE_STATE.get(device_id, {}).get("alert", "NORMAL")}

@app.post("/register_device")
async def register(data: dict):
    DEVICE_REGISTRY[data["device_id"]] = data["esp_ip"]
    DEVICE_STATE[data["device_id"]] = {
        "alert":"NORMAL","eye":None,"yawn":None,"last":0
    }
    return {"status":"registered"}

@app.post("/detect")
async def detect(file: UploadFile = File(...), device_id: str = ""):
    if device_id not in DEVICE_STATE:
        return {"alert":"NORMAL"}

    state = DEVICE_STATE[device_id]
    img = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    if img is None:
        return {"alert":"NORMAL"}

    res = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        state["eye"]=state["yawn"]=None
        state["alert"]="NORMAL"
        return {"alert":"NORMAL"}

    lm = res.multi_face_landmarks[0].landmark
    LEFT=[33,160,158,133,153,144]
    RIGHT=[362,385,387,263,373,380]

    e = (ear(lm,LEFT)+ear(lm,RIGHT))/2
    ref = dist((lm[33].x,lm[33].y),(lm[133].x,lm[133].y))
    mar = dist((lm[13].x,lm[13].y),(lm[14].x,lm[14].y))/ref

    now = time.time()

    if e < EYE_AR_THRESH:
        state["eye"] = state["eye"] or now
        if now - state["eye"] >= ALERT_DURATION and now - state["last"] > COOLDOWN:
            state["alert"]="EYE_CLOSED"
            state["last"]=now
            trigger_esp(device_id)
    else:
        state["eye"]=None

    if mar > MOUTH_AR_THRESH:
        state["yawn"] = state["yawn"] or now
        if now - state["yawn"] >= ALERT_DURATION and now - state["last"] > COOLDOWN:
            state["alert"]="YAWNING"
            state["last"]=now
            trigger_esp(device_id)
    else:
        state["yawn"]=None

    if state["alert"] not in ["EYE_CLOSED","YAWNING"]:
        state["alert"]="NORMAL"

    return {"alert":state["alert"]}
