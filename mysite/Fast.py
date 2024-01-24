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
    env_len: Optional[int] = 1


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


@app.get('/train')
async def root(background_tasks: BackgroundTasks):
    if websockets_connection is not None:
        global start_flag
        global first_init
        if start_flag:
            return {'message': '训练正在进行中'}
        background_tasks.add_task(train_model, websocket=websockets_connection)
        start_flag = True
        first_init = False
        return {'message': '启动成功！'}
    else:
        return {'message': '还没有建立websocket连接'}


@app.get('/test')
async def root(background_tasks: BackgroundTasks, step_num: int = 1):
    if websockets_connection is not None:
        global start_flag
        global first_init
        if start_flag:
            return {'message': '训练正在进行中'}
        background_tasks.add_task(test_model, websocket=websockets_connection, step_num=step_num)
        return {'message': '启动成功！'}
    else:
        return {'message': '还没有建立websocket连接'}


@app.post('/setting', response_model=Res)
async def create_setting(setting: Setting):
    global train
    if train is None:
        train = Train(setting.env_len, setting.processNum, setting.productNum)
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


async def train_model(websocket: WebSocket):
    global train
    global start_flag
    if train is not None:
        # 先训练
        batch_size = train.batch_size
        best_time = train.best_time
        max_steps = min(batch_size * 10, best_time * 5)
        for _ in range(max_steps):
            msg = train.step()
            await websocket.send_json(
                {'total_step': max_steps, 'step': msg['step'], 'progress': msg['step'] / max_steps})
        # 学习完成后
        start_flag = False


async def test_model(websocket: WebSocket, step_num: int = 1):
    global train
    if train is not None:
        # 再测试可视化
        train.reset()
        for _ in range(step_num):
            time.sleep(0.5)
            msg = train.test()
            await websocket.send_json(msg)
            if msg['dones'] == 1:
                break


if __name__ == '__main__':
    uvicorn.run("Fast:app", port=8080, reload=True)
