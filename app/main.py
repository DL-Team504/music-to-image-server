from diffusers.pipelines import stable_diffusion
from fastapi import FastAPI, File, Form, UploadFile, Depends
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import TypedDict, Any
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from diffusers import StableDiffusionPipeline

import librosa
import numpy as np
from sqlalchemy.orm import Session
from datetime import datetime
import soundfile as sf
import uuid

import dill


from app import crud, models, schemas
from .database import SessionLocal, engine
from .utils import createResponse, promptData
from .model import AudioCaptioningModel, Vocabulary, infer


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
    llm: PreTrainedModel
    llm_tokenizer: PreTrainedTokenizer
    stable_diffusion: StableDiffusionPipeline
    music_captioning: AudioCaptioningModel
    music_captioning_vocab: Vocabulary
    # music_captioning_vocab: Any


ml_models: MlModels = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("app/vocab-dill.pkl", "rb") as f:
        ml_models["music_captioning_vocab"] = dill.load(f)

    has_cuda = torch.cuda.is_available()
    stt_model_path = "ivrit-ai/whisper-large-v2-tuned"

    stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_path)
    if has_cuda:
        stt_model.to("cuda:0")

    ml_models["stt"] = stt_model
    ml_models["stt_processor"] = WhisperProcessor.from_pretrained(stt_model_path)
    ml_models["llm"] = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    ml_models["llm_tokenizer"] = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct"
    )
    ml_models["stable_diffusion"] = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    ).to("cuda")

    ml_models["music_captioning"] = AudioCaptioningModel(
        n_mels=128,
        vocab_size=len(ml_models["music_captioning_vocab"].itos),
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
    ).to("cuda")

    ml_models["music_captioning"].load_state_dict(torch.load("app\\best_model.pt"))

    yield

    ml_models.clear()


app = FastAPI(lifespan=lifespan)

app.mount("/images", StaticFiles(directory="images"), name="images")


@app.post("/generate-image")
def generate_image(
    start: float = Form(...),
    end: float = Form(...),
    image_style: str | None = Form(None),
    upload_file: UploadFile = File(...),
    db: Session = Depends(get_db)
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

    file_id = uuid.uuid4()

    sf.write(f'{file_id}.wav', audio, sr, subtype='PCM_24')

    lyrics = speech_to_text(audio, sr)
    captions = infer(
        f'{file_id}.wav',
        model=ml_models["music_captioning"],
        vocab=ml_models["music_captioning_vocab"],
        device="cuda",
    )

    image_description = createResponse(
        [
            {
                "role": "system",
                "content": "You are an expert in generating emotionally resonant and concise image descriptions.",
            },
            {"role": "user", "content": promptData(captions, lyrics)},
        ],
        ml_models["llm"],
        ml_models["llm_tokenizer"],
    )

    # Generate an image from a prompt
    image = ml_models["stable_diffusion"](image_description.split("\n")[-1]).images[0]
    title = upload_file.filename
    creation_date = datetime.today().strftime('%B %d, %Y')
    generated_image_create = schemas.GalleryImageCreate(path=f"images/{file_id}.png", title=title, creation_date=creation_date)
    crud.create_generated_image(db, generated_image_create)

    image.save(f"images/{file_id}.png")

    return speech_to_text(audio, sr)


@app.get("/gallery/images")
def get_gallery_images(db: Session = Depends(get_db)) -> list[schemas.GalleryImage]:
    images = crud.get_generated_images(db)

    return images
