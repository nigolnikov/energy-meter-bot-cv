import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import run_pipeline
from src.utils.logger import logger

app = FastAPI(title="Energy Meter Bot API")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/output")
def output():
    return FileResponse("output.jpg")


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_pipeline(tmp_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Cannot read file") from None
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail="Error in the pipeline") from e
    finally:
        os.remove(tmp_path)

    return result.to_dict()
