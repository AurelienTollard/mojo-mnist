import json

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, world!"}


@app.websocket("/ws")
async def ingest_images(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            meta_text = await websocket.receive_text()
            meta = json.loads(meta_text)
            width = int(meta["w"])
            height = int(meta["h"])
            payload = await websocket.receive_bytes()
            expected = width * height
            if len(payload) != expected:
                await websocket.send_json({"error": "invalid frame size"})
                continue
            array = np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
            mean_intensity = float(array.mean() / 255.0)
            probs = [1.0 - mean_intensity, mean_intensity]
            await websocket.send_json({"probs": probs})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
