import time

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel


class Setting(BaseModel):
    name: str
    age: int | None = None


class Res(BaseModel):
    message: str


app = FastAPI()
origins = ['http://localhost:8080']
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
websockets_connection = None


@app.get('/')
async def root(background_tasks: BackgroundTasks):
    if websockets_connection is not None:
        background_tasks.add_task(send_msg, msg='111', websocket=websockets_connection)
        return {'message': 'Hello World!'}
    else:
        return {'message': '还没有建立websocket连接'}


@app.post('/setting', response_model=Res)
async def create_setting(setting: Setting):
    return {'message': '1'}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, msg='000'):
    global websockets_connection
    websockets_connection = websocket
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}{msg}")
    except Exception as e:
        print(f"websocket connection disconnected{e}")
    finally:
        websockets_connection = None


async def send_msg(msg, websocket: WebSocket):
    try:
        time.sleep(3)
        await websocket.send_text(msg)
        print("send msg success")
    except Exception as e:
        print(f"websocket connection empty{e}")


if __name__ == '__main__':
    uvicorn.run("Fast:app", port=8080, reload=True)
