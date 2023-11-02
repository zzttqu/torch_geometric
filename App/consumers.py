import time

from channels.exceptions import StopConsumer
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer
import json
from model.train_class import Train
from loguru import logger


class TrainConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.train_handle: Train = None

    async def websocket_connect(self, message):
        """
        当有客户端向后端发送websocket连接请求时，自动触发该函数
        :param message:
        :return:
        """
        # 服务器允许客户端创建连接
        await self.accept()
        # self.train_handle = Train(2, 3)

    # 这个是处理所有请求的，不光包括websocket
    # def receive(self, text_data=None, bytes_data=None):
    #     super().receive(text_data, bytes_data)

    async def websocket_receive(self, message):
        """
        浏览器基于websocket向后端发送数据，自动触发接受消息，并且处理信息
        :param message:
        :return:
        """
        # 输出消息
        logger.info(message)
        data = message.get('text')
        data = json.loads(message['text'])
        json_str = json.dumps({'文字是': message['text']})
        if data['action'] == 'init' and self.train_handle is None:
            # 反复初始化Train会导致cell_id一直自增
            self.train_handle = Train(2, 3)
            json_str = json.dumps({'state': 'init success'})
            await self.send(text_data=json_str)
            # 只要开始，就不允许停止训练，必须完成所有step
            a = self.train_handle.train_online(10, False)
            for step_message in a:
                step_result = {'step': step_message[0], 'state': step_message[1]}
                await self.send(text_data=json.dumps(step_result))
            await self.send(text_data=json.dumps({'state': 'train success'}))
        elif data['action'] == 'init':
            a = self.train_handle.train_online(10, False)
            for step_message in a:
                step_result = {'step': step_message[0], 'state': step_message[1]}
                await self.send(text_data=json.dumps(step_result))
            await self.send(text_data=json.dumps({'state': 'train success'}))
        elif data['action'] == 'stop':
            pass
        # 服务端向前端回消息
        # await self.send(text_data=json_str)

    async def websocket_disconnect(self, message):
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
