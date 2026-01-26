import asyncio

import numpy as np
import websockets


async def main() -> None:
    data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    async with websockets.connect("ws://127.0.0.1:8000/ws") as websocket:
        await websocket.send(data.tobytes())
        reply = await websocket.recv()
        print(reply)


if __name__ == "__main__":
    asyncio.run(main())
