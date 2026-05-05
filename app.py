from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import torch
import numpy as np
from PIL import Image

from model import MultimodalAutismModel
from eye_tracking import get_eye_contact_score

# ---------------------------------------
# INIT APP
# ---------------------------------------

app = FastAPI()

# ---------------------------------------
# CORS
# ---------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"

# ---------------------------------------
# LOAD MODEL
# ---------------------------------------

print("🚀 Loading model...")

model = MultimodalAutismModel().to(DEVICE)
model.load_state_dict(torch.load("autism_model.pt", map_location=DEVICE))
model.eval()

print("✅ Model loaded successfully")

# ---------------------------------------
# SERVE FRONTEND
# ---------------------------------------

@app.get("/")
def home():
    print("🔥 FRONTEND SERVED")
    return FileResponse("static/index.html")

# ---------------------------------------
# IMAGE PREPROCESS
# ---------------------------------------

def preprocess(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
    return img

# ---------------------------------------
# PREDICTION API
# ---------------------------------------

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age_months: int = Form(...),
    gender: int = Form(...),
    jaundice: int = Form(...),
    family_autism: int = Form(...),
    A1:int = Form(...), A2:int = Form(...), A3:int = Form(...),
    A4:int = Form(...), A5:int = Form(...), A6:int = Form(...),
    A7:int = Form(...), A8:int = Form(...), A9:int = Form(...),
    A10:int = Form(...),
    heart_rate: float = Form(...),
    temperature: float = Form(...),
    use_eye_tracking: bool = Form(False)
):

    print("📥 Prediction request received")

    # ---------------------------------------
    # IMAGE
    # ---------------------------------------

    img = preprocess(image.file)

    # ---------------------------------------
    # ADOS (behavior)
    # ---------------------------------------

    ados = torch.tensor([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,age_months,gender]]).float()

    # ---------------------------------------
    # PHYSIO
    # ---------------------------------------

    physio = torch.tensor([[
        [heart_rate,0.3,temperature,98]
    ]*5]).float()

    # ---------------------------------------
    # DUMMY INPUTS
    # ---------------------------------------

    aud = torch.zeros((1,40,100))
    mot = torch.zeros((1,7,900))

    # ---------------------------------------
    # OPTIONAL EYE TRACKING
    # ---------------------------------------

    if use_eye_tracking:
        try:
            eye_score = get_eye_contact_score()
        except:
            eye_score = None
    else:
        eye_score = None

    # ---------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------

    with torch.no_grad():
        out = model(img,aud,mot,physio,ados)
        prob = torch.softmax(out,dim=1)[0][1].item()

    # ---------------------------------------
    # COUNT ATYPICAL ANSWERS
    # ---------------------------------------

    atypical_count = sum([A1,A2,A3,A4,A5,A6,A7,A8,A9,A10])

    # ---------------------------------------
    # RISK LOGIC (BASED ON ANSWERS)
    # ---------------------------------------

    if atypical_count >= 6:
        risk = "High"

    elif atypical_count >= 3:
        risk = "Moderate"

    else:
        risk = "Low"

    print(f"✅ Result: {risk}, Prob: {prob:.3f}, Atypical: {atypical_count}")

    # ---------------------------------------
    # RESPONSE
    # ---------------------------------------

    return {
        "Autism_Risk": risk,
        "Model_Probability": round(prob,3),
        "Atypical_Count": atypical_count,
        "Eye_Contact_Score": eye_score
    }