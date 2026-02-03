from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from .cnn import CNN, SimpleCNN
from .mlp import HIDDEN_SIZE, MLP, MojoMLP, get_device
from PIL import Image

app = FastAPI()

DEFAULT_MLP_PATH = Path(__file__).parent.parent / ".data" / "mlp_mnist.pth"
DEFAULT_CNN_PATH = Path(__file__).parent.parent / ".data" / "cnn_mnist.pth"
DEMO_DIR = Path(__file__).parent
INDEX_FILE = DEMO_DIR / "index.html"
MODEL_CACHE: dict[tuple[str, str, str, str], torch.nn.Module] = {}


def _load_model(
    backend: str, model_type: str, model_path: Path, device: torch.device
) -> torch.nn.Module:
    cache_key = (backend, model_type, str(model_path), device.type)
    cached = MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not model_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Model file not found: {model_path}",
        )

    state_dict = torch.load(model_path, map_location="cpu")

    if model_type == "mlp":
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
    elif model_type == "cnn":
        if backend == "pytorch":
            model = CNN().to(device)
            model.load_state_dict(state_dict)
        elif backend == "mojo":
            if device.type != "cuda":
                raise HTTPException(
                    status_code=400,
                    detail="Mojo backend requires CUDA",
                )
            model = SimpleCNN().to(device)
            model.load_from_state_dict(state_dict)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown backend: {backend}",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_type: {model_type}",
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

    return arr


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
    model_type: str = Query("mlp", description="Model type: mlp or cnn"),
    model: str | None = Query(None, description="Path to model .pth"),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    device = get_device()
    if model is None:
        model_path = DEFAULT_MLP_PATH if model_type == "mlp" else DEFAULT_CNN_PATH
    else:
        model_path = Path(model)

    model_obj = _load_model(backend, model_type, model_path, device)

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
    if model_type == "cnn":
        inputs = torch.from_numpy(batch[:, None, :, :]).to(device)
    else:
        inputs = torch.from_numpy(batch.reshape(batch.shape[0], -1)).to(device)

    with torch.no_grad():
        logits = model_obj(inputs)
        outputs = torch.softmax(logits, dim=1)

    outputs_cpu = outputs.detach().cpu().numpy()
    predictions = outputs_cpu.argmax(axis=1).tolist()

    return {
        "backend": backend,
        "model_type": model_type,
        "model": str(model_path),
        "batch_size": int(outputs_cpu.shape[0]),
        "outputs": outputs_cpu.tolist(),
        "predictions": predictions,
    }


if __name__ == "__main__":
    if __package__ in (None, ""):
        raise RuntimeError("Run as a module: python -m demo.server")
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
