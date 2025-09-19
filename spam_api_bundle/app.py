from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle, os

BASE_DIR = os.path.dirname(__file__)

app = FastAPI(title="Spam Detector API")

# serve static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# load model and vectorizer at startup
MODEL_PATH = os.path.join(BASE_DIR, r"D:\Buildables Internship\Cyber_Bullying\spam_api_bundle\vectorizer.pkl")
CLF_PATH = os.path.join(BASE_DIR, r"D:\Buildables Internship\Cyber_Bullying\spam_api_bundle\model.pkl")

def load_pickles(vectorizer_path, model_path):
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

# provide a helpful error if files not present
try:
    vectorizer, model = load_pickles(MODEL_PATH, CLF_PATH)
except Exception as e:
    vectorizer, model = None, None
    load_error = str(e)
else:
    load_error = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "load_error": load_error})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, message: str = Form(...)):
    if vectorizer is None or model is None:
        return templates.TemplateResponse("index.html", {"request": request, "load_error": load_error})
    X = vectorizer.transform([message])
    pred = model.predict(X)[0]
    prob = max(model.predict_proba(X)[0]) if hasattr(model, "predict_proba") else None
    label = "Spam" if pred==1 else "Ham"
    return templates.TemplateResponse("index.html", {"request": request, "message": message, "label": label, "prob": prob, "load_error": load_error})

@app.post("/api/predict", response_class=JSONResponse)
async def api_predict(payload: dict):
    text = payload.get("text","")
    if vectorizer is None or model is None:
        return JSONResponse({"error": "Model not loaded. Place vectorizer.pkl and model.pkl in app folder."}, status_code=500)
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = max(model.predict_proba(X)[0]) if hasattr(model, "predict_proba") else None
    return {"prediction": int(pred), "label": "spam" if pred==1 else "ham", "probability": float(prob) if prob is not None else None}
