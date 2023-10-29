from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core import serializers
from model.train_class import Train


# Create your views here.
def index(request):
    serial_data = serializers.serialize("json", {'name': 'zzz'})
    # return HttpResponse('你是谁？')
    return JsonResponse(serial_data)


def train(request):
    train_handle = Train(2, 3)
    train_handle.train_online(1)
