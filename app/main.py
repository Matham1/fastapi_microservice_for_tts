from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from app.tts import TTSService

class TTSRequest(BaseModel):
    text: str

app = FastAPI()
tts_service: TTSService

@app.on_event("startup")
async def load_model():
    # point to your config & checkpoint files
    config_path = "/app/vits2_inference/configs/vits2_config.json"
    ckpt_path   = "/app/vits2_inference/checkpoints/vits2_ckpt.pth"
    global tts_service
    tts_service = TTSService(config_path, ckpt_path)

@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    wav_bytes = tts_service.synthesize(req.text)
    return StreamingResponse(io.BytesIO(wav_bytes),
                             media_type="audio/wav",
                             headers={"Content-Disposition": "inline; filename=tts.wav"})
