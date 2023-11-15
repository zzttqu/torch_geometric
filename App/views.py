import asyncio
import time
from functools import partial
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.core import serializers
from loguru import logger
from django.views.decorators.csrf import csrf_exempt
from .consumers import TrainConsumer


# Create your views here.
# 问题就是，如果用http发送信息的，然后websocket传输的话，会有一点麻烦，因为websocket是有状态的，需要先建立连接才能发送数据
async def index(request: HttpRequest):
    a = {'name': 'zzz'}
    # loop = asyncio.get_event_loop()
    # partial_f = partial(sleep, loop=loop)
    # await loop.run_in_executor(None, sleep)
    # 实际最好用的异步方法
    # asyncio.create_task(sync_sleep(2, 2, 64))
    # loop = asyncio.get_event_loop()
    # loop.create_task(sleep())

    return HttpResponse('async view call async request')


async def sync_sleep(func_num, center_num, step_num):
    loop = asyncio.get_event_loop()
    partial_f = partial(init_train, func_num=func_num, center_num=center_num, step_num=step_num)
    await loop.run_in_executor(None, partial_f)


def init_train(func_num, center_num, step_num):
    train_consumer = TrainConsumer()
    train_consumer.init_train(func_num, center_num, step_num)

# def train(request):
#     a = {'state': '初始化成功'}
#     obj = JsonResponse(a, safe=False)
#     # obj['Access-Control-Allow-Origin'] = 'http://localhost:8080'
#     return obj
