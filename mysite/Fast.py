import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel

from train_class import Train


class Setting(BaseModel):
    processNum: int = 5
    productNum: int = 5


class Res(BaseModel):
    message: str


app = FastAPI()
origins = ['http://localhost:3000']
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
websockets_connection = None
train: Optional[Train] = None
start_flag = False
first_init = True


@app.get('/start')
async def root(background_tasks: BackgroundTasks, step_num: int = 1):
    if websockets_connection is not None:
        global start_flag
        global first_init
        if start_flag:
            return {'message': '训练正在进行中'}
        background_tasks.add_task(send_msg, websocket=websockets_connection, step_num=step_num)
        start_flag = True
        first_init = False
        return {'message': '启动成功！'}
    else:
        return {'message': '还没有建立websocket连接'}


@app.post('/setting', response_model=Res)
async def create_setting(setting: Setting):
    global train
    if train is None:
        train = Train(1, setting.processNum, setting.productNum)
        return {'message': '1'}
    else:
        return {'message': '0'}


@app.get('/select_env')
async def create_setting(index: int = 0):
    global train
    if train is not None:
        env_init_info = train.init_setting(index, first_init)
        return env_init_info
    else:
        return {'message': '0'}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global websockets_connection
    websockets_connection = websocket
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        print(f"websocket connection disconnected{e}")
    finally:
        websockets_connection = None


async def send_msg(websocket: WebSocket, step_num: int = 1):
    global train
    global start_flag
    if train is not None:
        for _ in range(step_num):
            time.sleep(0.5)
            msg = train.step(step_num)
            await websocket.send_json(msg)
        # 循环完成后
        start_flag = False


if __name__ == '__main__':
    uvicorn.run("Fast:app", port=8080, reload=True)
