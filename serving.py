import logging
from pathlib import Path
from typing import Literal, Optional

import fastapi
import fastapi.middleware.cors
import tyro
import uvicorn
from attr import dataclass
from fastapi import Form, HTTPException, status
from fastapi.responses import Response

from fam.llm.fast_inference import TTS

logger = logging.getLogger(__name__)
app = fastapi.FastAPI()


@dataclass
class ServingConfig:
    huggingface_repo_id: str = "metavoiceio/metavoice-1B-v0.1"
    """Absolute path to the model directory."""

    temperature: float = 1.0
    """Temperature for sampling applied to both models."""

    seed: int = 1337
    """Random seed for sampling."""

    port: int = 58003

    quantisation_mode: Optional[Literal["int4", "int8"]] = None


# Singleton
class _GlobalState:
    config: ServingConfig
    tts: TTS


GlobalState = _GlobalState()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/tts", response_class=Response)
async def text_to_speech(
    text: str = Form(None, description="Text to convert to speech."),
    speaker_ref_path: Optional[str] = Form("https://cdn.themetavoice.xyz/speakers/bria.mp3", description="URL or path to an audio file of a reference speaker."),
    guidance: float = Form(3.0, description="Control speaker similarity - how closely to match speaker identity and speech style, range: 0.0 to 5.0.", ge=0.0, le=5.0),
    top_p: float = Form(0.95, description="Controls speech stability - improves text following for a challenging speaker, range: 0.0 to 1.0.", ge=0.0, le=1.0),
):
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text to convert into speech must be provided.",
        )

    wav_out_path = None

    try:
        if not speaker_ref_path:
            logger.warn("Running without speaker reference! Guidance will not be used.")
            guidance = None

        wav_out_path = GlobalState.tts.synthesise(
            text=text,
            spk_ref_path=speaker_ref_path,
            top_p=top_p,
            guidance_scale=guidance,
        )

        with open(wav_out_path, "rb") as f:
            return Response(content=f.read(), media_type="audio/wav")
    except Exception:
        logger.exception(
            f"Error processing request. text: {text}, speaker_ref_path: {speaker_ref_path}, guidance: {guidance}, top_p: {top_p}"
        )
        return Response(
            content="Something went wrong. Please try again in a few mins or contact us on Discord.",
            status_code=500,
        )
    finally:
        if wav_out_path is not None:
            Path(wav_out_path).unlink(missing_ok=True)


if __name__ == "__main__":
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

    logging.root.setLevel(logging.INFO)

    GlobalState.config = tyro.cli(ServingConfig)
    GlobalState.tts = TTS(
        seed=GlobalState.config.seed,
        quantisation_mode=GlobalState.config.quantisation_mode,
        telemetry_origin="api_server",
    )

    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*", f"http://localhost:{GlobalState.config.port}", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=GlobalState.config.port,
        log_level="info",
    )
