from django.urls import re_path, path
from App.consumers import TrainConsumer

websocket_urlpatterns = [
    path('room/', TrainConsumer.as_asgi()),
    # 也可以使用正则路径,这种方式用在群组通信当中
    # re_path('ws/chat/(?P<group>\w+)/$', ChatConsumer)
]
