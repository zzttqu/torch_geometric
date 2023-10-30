from channels.exceptions import StopConsumer
from channels.generic.websocket import WebsocketConsumer
import json
from model.train_class import Train
from loguru import logger


class TrainConsumer(WebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.train_handle: Train = None

    def websocket_connect(self, message):
        """
        当有客户端向后端发送websocket连接请求时，自动触发该函数
        :param message:
        :return:
        """
        # 服务器允许客户端创建连接
        self.accept()
        del self.train_handle
        self.train_handle = Train(2, 3)

    def websocket_receive(self, message):
        """
        浏览器基于websocket向后端发送数据，自动触发接受消息，并且处理信息
        :param message:
        :return:
        """
        # 输出消息
        json_str = json.dumps({'文字是': message['text']})
        if message['text'] == '1':
            a = self.train_handle.train_online(False)
            # logger.info(a)
            # 我觉得可以处理一下，比如
            # a = {'state': '训练了一步成功'}
            json_str = json.dumps(a)
        # 服务端向前端回消息
        self.send(text_data=json_str)

    def websocket_disconnect(self, message):
        """
        客户端与服务端断开连接时，自动触发该函数
        :param message:
        :return:
        """
        print('断开连接，保存模型')
        if self.train_handle is not None:
            self.train_handle.train_online(True)
            del self.train_handle
        raise StopConsumer()
