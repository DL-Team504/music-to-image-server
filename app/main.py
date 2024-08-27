from fastapi import FastAPI, File, Form, UploadFile, Depends
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import TypedDict
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
from sqlalchemy.orm import Session
from datetime import datetime

from . import crud, models, schemas
from .database import SessionLocal, engine


from transformers.modeling_utils import PreTrainedModel


models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def speech_to_text(audio: np.ndarray, sr: int) -> str:
    SAMPLING_RATE = 16000

    audio_resample = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    input_features = ml_models["stt_processor"](
        audio_resample, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features

    if torch.cuda.is_available():
        input_features = input_features.to("cuda:0")

    predicted_ids = ml_models["stt"].generate(
        input_features, language="he", num_beams=5
    )
    transcript = ml_models["stt_processor"].batch_decode(
        predicted_ids, skip_special_tokens=True
    )

    return transcript[0]


class MlModels(TypedDict):
    stt: PreTrainedModel
    stt_processor: WhisperProcessor


ml_models: MlModels = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    has_cuda = torch.cuda.is_available()
    stt_model_path = "ivrit-ai/whisper-large-v2-tuned"

    stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_path)
    if has_cuda:
        stt_model.to("cuda:0")

    ml_models["stt"] = stt_model
    ml_models["stt_processor"] = WhisperProcessor.from_pretrained(stt_model_path)

    yield

    ml_models.clear()


app = FastAPI(lifespan=lifespan)

app.mount("/images", StaticFiles(directory="images"), name="images")


@app.post("/generate-image")
def generate_image(
    start: int = Form(...),
    end: int = Form(...),
    image_style: str | None = Form(None),
    upload_file: UploadFile = File(...),
) -> str:
    # splice audio according to focus_area
    # pass to audio to image function
    # save the image received from the function
    # make a db record of {image_url, title(audio file name), creation_date}
    # return the url location of the image

    if start < 0:
        start = 0

    end = min(start + 30, end)

    duration = end - start

    audio, sr = librosa.load(upload_file.file, offset=start, duration=duration)

    audio_text = speech_to_text(audio, sr)

    # image = generate_image(audio_text, audio)
    # path = save_image(image)
    # title = upload_file.filename
    # creation_date = datetime.today().strftime('%B %d, %Y')
    # generated_image_create = schemas.GalleryImageCreate(path=path, title=title, creation_date=creation_date)
    # crud.create_generated_image(db, generated_image_create)
    # return path

    return speech_to_text(audio, sr)


@app.get("/gallery/images")
def get_gallery_images(db: Session = Depends(get_db)) -> list[schemas.GalleryImage]:
    images = crud.get_generated_images(db)

    return images
