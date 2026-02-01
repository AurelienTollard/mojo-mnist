from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from mlp import HIDDEN_SIZE, MLP, MojoMLP, get_device

try:
    from PIL import Image, ImageOps
except Exception as exc:  # pragma: no cover - optional dependency
    Image = None
    ImageOps = None
    _pil_import_error = exc

app = FastAPI()

DEFAULT_MODEL_PATH = Path(__file__).parent / ".data" / "mlp_mnist.pth"
STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}


def _ensure_pil() -> None:
    if Image is None or ImageOps is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pillow is required to decode images. Install 'pillow' "
                "or send raw 28x28 grayscale bytes."
            ),
        ) from _pil_import_error


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
    _ensure_pil()
    image_module = cast(Any, Image)
    image_ops = cast(Any, ImageOps)

    with image_module.open(io.BytesIO(contents)) as img:
        img = img.convert("L")
        # Invert to match MNIST (white digits on black background)
        img = image_ops.invert(img)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    coords = np.argwhere(arr > 0.05)
    if coords.size == 0:
        return np.zeros((28 * 28,), dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    crop = arr[y0:y1, x0:x1]
    height, width = crop.shape

    if height == 0 or width == 0:
        return np.zeros((28 * 28,), dtype=np.float32)

    resample = (
        image_module.Resampling.LANCZOS
        if hasattr(image_module, "Resampling")
        else image_module.LANCZOS
    )

    scale = 20.0 / float(max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    crop_img = image_module.fromarray((crop * 255).astype(np.uint8))
    crop_img = crop_img.resize((new_w, new_h), resample=resample)
    crop_arr = np.asarray(crop_img, dtype=np.float32) / 255.0

    pad_y = 28 - new_h
    pad_x = 28 - new_w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    padded = np.pad(crop_arr, ((top, bottom), (left, right)), mode="constant")
    if padded.shape != (28, 28):
        padded = (
            np.asarray(crop_img.resize((28, 28), resample=resample), dtype=np.float32)
            / 255.0
        )

    return padded.reshape(-1)


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
