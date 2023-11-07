import asyncio
import time

import torch
from channels.exceptions import StopConsumer
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer
import json
from model.train_class import Train
from loguru import logger


class TrainConsumer(WebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.train_handle: Train | None = None

    def websocket_connect(self, message):
        """
        当有客户端向后端发送websocket连接请求时，自动触发该函数
        :param message:
        :return:
        """
        # 服务器允许客户端创建连接
        self.accept()
        # self.train_handle = Train(2, 3)

    # 这个是处理所有请求的，不光包括websocket
    # def receive(self, text_data=None, bytes_data=None):
    #     super().receive(text_data, bytes_data)

    def websocket_receive(self, message):
        """
        浏览器基于websocket向后端发送数据，自动触发接受消息，并且处理信息
        :param message:
        :return:
        """
        # 输出消息
        # data = message.get('text')
        data = json.loads(message['text'])
        if data['action'] == 'init' and self.train_handle is None:
            func_num = 2
            center_num = 2
            step_num = 10
            # 判断是否发送过来了这两个键
            if 'func_num' in data:
                func_num = int(data['func_num'])
            if 'center_num' in data:
                center_num = int(data['center_num'])
            if 'step_num' in data:
                step_num = int(data['step_num'])
            # 反复初始化Train会导致cell_id一直自增
            self.train_handle = Train(func_num, center_num, load_model=False)
            json_str = json.dumps({'status': 'init success'})
            self.send(text_data=json_str)
            # 只要开始，就不允许停止训练，必须完成所有step
            # 这是一个生成器
            a = self.train_handle.train_online(step_num, False)
            for step_message in a:
                if step_message == 'training':
                    self.send(text_data=json.dumps({'status': 'training'}))
                else:
                    # 慢一点看效果
                    time.sleep(0.5)
                    step_result = {'status': 'running', 'step': step_message[0], 'state': step_message[1]}
                    self.send(text_data=json.dumps(step_result))
            # 迭代完成后
            self.send(text_data=json.dumps({'status': 'finish'}))
            del self.train_handle
            torch.cuda.empty_cache()
        elif data['action'] == 'init' and self.train_handle is not None:
            self.send(text_data=json.dumps({'status': 'init failed'}))
        elif data['action'] == 'stop':
            pass
        # 服务端向前端回消息
        # await self.send(text_data=json_str)

    def websocket_disconnect(self, message):
        """
        客户端与服务端断开连接时，自动触发该函数
        :param message:
        :return:
        """
        self.train_handle = None
        print('断开连接')
        # if self.train_handle is not None:
        #     self.train_handle.train_online(True)
        raise StopConsumer()
