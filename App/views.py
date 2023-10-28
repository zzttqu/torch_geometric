from django.shortcuts import render
from django.http import HttpResponse, JsonResponse


# Create your views here.
def index(request):
    # return HttpResponse('你是谁？')
    return JsonResponse({'name': 'zzz'})
