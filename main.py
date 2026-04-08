"""
Cloud API - receives JPEG frames from the dge agenet, clasifies them with
Gemini Vision (Vertex AI), and fires a Pushover notificarion if a parking
ranger is detected.

Environment variables (set in Render dashboard):
    CLOUD_API_KEY           Shared secret the edge agent send in X-API-Key header
    GCP_PROJECT_ID          Google Cloud project ID
    GCP_REGION              Vertex AI region
    GCP_CREDENTIALS_B64     Base64-encoded service account JSON key
    PUSHOVER_APP_TOKEN      Pushover application token
    PUSHOVER_USER_KEY       Pushover user/group key for the client's device
    CONFIDENCE_THRESHOLD    Float 0-1, default 0.75
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

import httpx
import vertexai
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from vertexai.generative_models import GenerativeModel, Image

logging.basicConfig(level=logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# -- Config -------------------------------------------------------------------------------

CLOUD_API_KEY           = os.environ["CLOUD_API_KEY"]
GCP_PROJECT_ID          = os.environ["GCP_PROJECT_ID"]
GCP_REGION              = os.environ.get("GCP_REGION", "us-central1")
GCP_CREDENTIALS_B64     = os.environ["GCP_CREDENTIALS_B64"]
PUSHOVER_APP_TOKEN      = os.environ["PUSHOVER_APP_TOKEN"]
PUSHOVER_USER_KEY       = os.environ["PUSHOVER_USER_KEY"]
CONFIDENCE_THRESHOLD    = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))

# Cooldown: don't spam notifications if the ranger stays in frame
NOTIFICATION_COOLDOWN_SECONDS = int(os.environ.get("NOTIFICATION_COOLDOWN_SECONDS", "120"))
_last_notification_time: float = 0.0

# Initialised during startup lifespan - not at import time
gemini: GenerativeModel | None = None

# -- Lifespan -------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_):
    global gemini
    # Decode serviuce account JSON and point the SDK at it
    creds_json = base64.b64decode(GCP_CREDENTIALS_B64)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(creds_json)
    tmp.flush()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
    log.info("GCP credentials written to %s", tmp.name)


    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    gemini = GenerativeModel("gemini-2.5-pro")
    log.info("Vertex AI initialised (project=%s region=%s)", GCP_PROJECT_ID, GCP_REGION)

    yield
    log.info("Shutting down")

app = FastAPI(title = "CCTV Parking Ranger Classifier", lifespan = lifespan)

# -- Auth -----------------------------------------------------------------------------------

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != CLOUD_API_KEY:
        raise HTTPException(status_code=401, detail = "Invalid API Key")

# -- Gemini Vision classifier ---------------------------------------------------------------

PROMPT = """\
You are a computer vision assistant analysing CCTV footage for a small business.
Your sole task is to determine whether a parking ranger (council parking officer) is visible in the image.

Parking rangers typically wear:
- High visibility vests or jackets (fluro yellow, green)
- Council or local government uniforms
- They may be carrying handheld ticketing devices or looking at vehicles

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{"ranger_detected": tru/false, "confidence": 0.0-1.0, "reason": "brief explanation"}

"""


async def classify_frame(jpeg_bytes: bytes) -> dict:
    image = Image.from_bytes(jpeg_bytes)
    # vertexai SDK is synchronous - run in a thread so we don't block the event loop
    response = await asyncio.to_thread(
        gemini.generate_content,
        [image, PROMPT],
        generation_config={"max_output_tokens": 3000, "temperature": 0}
    )
    log.info(f"This is the response: {response}")
    # Robustly extract text from different GenerationResponse shapes
    if hasattr(response, "text"):
        text = response.text.strip()
    elif hasattr(response, "output_text"):
        text = response.output_text.strip()
    elif hasattr(response, "generations") and response.generations:
        gen0 = response.generations[0]
        text = getattr(gen0, "text", str(gen0)).strip()
    else:
        text = str(response).strip()
        
    # Strip accidental mardown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text)

# -- Pushover notification -------------------------------------------------------------------

async def send_pushover(reason: str, confidence: float, jpeg_bytes: bytes | None = None):
    global _last_notification_time
    now = time.monotonic()
    if now - _last_notification_time < NOTIFICATION_COOLDOWN_SECONDS:
        log.info("Notification suppressed (cooldown active)")
        return
    async with httpx.AsyncClient() as client:
        # Pushover supports a single file attachment under the 'attachment' multipart field.
        # If we have JPEG bytes, send as multipart/form-data with files; otherwise send as form data.
        url = "https://api.pushover.net/1/messages.json"
        common_data = {
            "token":    PUSHOVER_APP_TOKEN,
            "user":     PUSHOVER_USER_KEY,
            "title":    "Parking Ranger Spotted!",
            "message":  f"A parking ranger has been detected outside ({confidence:.0%} confidence). {reason}",
            "sound":    "siren",
            "priority": 1, # high priority - bypasses quiet hours
        }

        if jpeg_bytes:
            files = {
                # filename can be anything; ensure correct content-type
                "attachment": ("frame.jpg", jpeg_bytes, "image/jpeg")
            }
            resp = await client.post(url, data=common_data, files=files, timeout=10)
        else:
            resp = await client.post(url, data=common_data, timeout=10)
    resp.raise_for_status()
    _last_notification_time = now
    log.info("Pushover notification sent (confidence=%.2f)", confidence)


# -- Routes ---------------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
):
    verify_api_key(x_api_key)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code = 400, detail = "File must be an image")
    
    jpeg_bytes = await file.read()
    if len(jpeg_bytes) > 10 * 1024 * 1024: #10 MB safety limit
        raise HTTPException(status_code=413, detail = "Image too large (max 10 MB)")
    log.info("Classifying frame (%d KB)", len(jpeg_bytes) // 1024)

    try:
        result = await classify_frame(jpeg_bytes)
    except Exception as exc:
        log.exception("Classification failed: %s", exc)
        raise HTTPException(status_code = 502, detail=f"Classification error: {exc}")
    
    ranger_detected = result.get("ranger_detected", False)
    confidence = float(result.get("confidence", 0.0))
    reason = result.get("reason", "")

    log.info("Result: detected=%s confidence=%.2f reason=%s", ranger_detected, confidence, reason)

    notification_sent = False
    if ranger_detected and confidence >= CONFIDENCE_THRESHOLD:
        try:
            # Pass the original JPEG bytes so the device receives the image as an attachment
            await send_pushover(reason, confidence, jpeg_bytes)
            notification_sent = True
        except Exception as exc:
            log.error("Pushover failed: %s", exc)
    
    return JSONResponse({
        "ranger_detected":      ranger_detected,
        "confidence":           confidence,
        "reason":               reason,
        "notification_sent":    notification_sent,
        "threshold":            CONFIDENCE_THRESHOLD,

    })