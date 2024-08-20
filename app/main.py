from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
from typing import BinaryIO, TypedDict
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


from transformers.modeling_utils import PreTrainedModel


def to_camel_case(string: str) -> str:
    res = "".join(word.capitalize() for word in string.split("_"))
    return res[0].lower() + res[1:]


def speech_to_text(audio: BinaryIO) -> str:
    SAMPLING_RATE = 16000
    audio, sr = librosa.load(audio)

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


class GalleryImage(BaseModel):
    img_url: str
    title: str
    creation_date: str

    model_config = ConfigDict(alias_generator=to_camel_case, populate_by_name=True)


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


# mount director where generated images are stored


@app.post("/generate-image")
def generate_image(
    start: int = Form(...),
    end: int = Form(...),
    image_style: str | None = Form(None),
    audio: UploadFile = File(...),
) -> str:
    # load audio file
    # if needed clamp focus_area to audio length
    # splice audio according to focus_area
    # pass to audio to image function
    # save the image received from the function
    # make a db record of {image_url, title(audio file name), creation_date}
    # return the url location of the image

    return speech_to_text(audio.file)


@app.get("/gallery/images")
def get_gallery_images() -> list[GalleryImage]:
    # return a list of the last ~20 records in the db
    return
