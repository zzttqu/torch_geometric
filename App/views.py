from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.core import serializers
from model.train_class import Train
from loguru import logger


# Create your views here.


class ModelT:
    def __init__(self):
        self.train_handle: Train = None

    def index(self, request: HttpRequest):
        a = {'name': 'zzz'}
        logger.debug(request.method)
        # return HttpResponse('你是谁？')
        return JsonResponse(a)

    def init_train(self, request):
        self.train_handle = Train(2, 3)
        a = {'state': '初始化成功'}
        return JsonResponse(a)

    def train(self, request):
        if self.train_handle is None:
            self.train_handle = Train(2, 3)
        a = self.train_handle.train_online(1)
        logger.info(a)
        # 我觉得可以处理一下，比如
        # a = {'state': '训练了一步成功'}
        return JsonResponse(a, safe=False)
