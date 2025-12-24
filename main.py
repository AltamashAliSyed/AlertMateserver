from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2, numpy as np, mediapipe as mp, math, time

app = FastAPI()

CURRENT_ALERT = "NORMAL"
LAST_UPDATE = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.45


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


@app.get("/")
def root():
    return {
        "status": "AlertMate API is running",
        "detect_endpoint": "/detect",
        "status_endpoint": "/status"
    }


@app.get("/status")
def get_status():
    global CURRENT_ALERT
    if time.time() - LAST_UPDATE > 5:
        CURRENT_ALERT = "NORMAL"
    return {"alert": CURRENT_ALERT}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global CURRENT_ALERT, LAST_UPDATE

    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        CURRENT_ALERT = "NORMAL"
        return {"alert": CURRENT_ALERT}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        CURRENT_ALERT = "NORMAL"
        return {"alert": CURRENT_ALERT}

    lm = result.multi_face_landmarks[0].landmark

    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    ear = (compute_ear(lm, LEFT_EYE) + compute_ear(lm, RIGHT_EYE)) / 2

    ref = euclidean(
        (lm[LEFT_EYE[0]].x, lm[LEFT_EYE[0]].y),
        (lm[LEFT_EYE[3]].x, lm[LEFT_EYE[3]].y)
    )

    mar = mouth_ratio(lm, ref=ref)

    if ear < EYE_AR_THRESH:
        CURRENT_ALERT = "EYE_CLOSED"
    elif mar > MOUTH_AR_THRESH:
        CURRENT_ALERT = "YAWNING"
    else:
        CURRENT_ALERT = "NORMAL"

    LAST_UPDATE = time.time()
    return {"alert": CURRENT_ALERT}
