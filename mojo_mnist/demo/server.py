from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from mojo_mnist.demo.mlp import HIDDEN_SIZE, MLP, MojoMLP, get_device

app = FastAPI()

DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / ".data" / "mlp_mnist.pth"
DEMO_DIR = Path(__file__).parent
INDEX_FILE = DEMO_DIR / "index.html"
MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}


def _load_model(
    backend: str, model_path: Path, device: torch.device
) -> torch.nn.Module:
    cache_key = (backend, str(model_path), device.type)
    cached = MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not model_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Model file not found: {model_path}",
        )

    state_dict = torch.load(model_path, map_location="cpu")

    if backend == "pytorch":
        model = MLP(hidden_size=HIDDEN_SIZE)
        model.load_state_dict(state_dict)
        model = model.to(device)
    elif backend == "mojo":
        if device.type != "cuda":
            raise HTTPException(
                status_code=400,
                detail="Mojo backend requires CUDA",
            )
        model = MojoMLP(hidden_size=HIDDEN_SIZE).to(device)
        model.load_from_state_dict(state_dict)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {backend}",
        )

    model.eval()
    MODEL_CACHE[cache_key] = model
    return model


def _decode_image(contents: bytes) -> np.ndarray:
    image_module = cast(Any, Image)

    with image_module.open(io.BytesIO(contents)) as img:
        img = img.convert("L")
        if img.size != (28, 28):
            raise HTTPException(
                status_code=400,
                detail=(f"Invalid image size {img.size}. Expected 28x28 pixels."),
            )
        arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr.reshape(-1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Client page not found. Expected static/index.html",
        )
    return FileResponse(INDEX_FILE)


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    backend: str = Query("mojo", description="Backend: mojo or pytorch"),
    model: str = Query(str(DEFAULT_MODEL_PATH), description="Path to model .pth"),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    device = get_device()
    model_path = Path(model)
    model_obj = _load_model(backend, model_path, device)

    images: list[np.ndarray] = []
    for upload in files:
        contents = await upload.read()
        try:
            images.append(_decode_image(contents))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode image '{upload.filename}': {exc}",
            ) from exc

    batch = np.stack(images, axis=0)
    inputs = torch.from_numpy(batch).to(device)

    with torch.no_grad():
        logits = model_obj(inputs)
        outputs = torch.softmax(logits, dim=1)

    outputs_cpu = outputs.detach().cpu().numpy()
    predictions = outputs_cpu.argmax(axis=1).tolist()

    return {
        "backend": backend,
        "model": str(model_path),
        "batch_size": int(outputs_cpu.shape[0]),
        "outputs": outputs_cpu.tolist(),
        "predictions": predictions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
