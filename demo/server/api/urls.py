from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('split-slider', views.split_slider, name='split-slider'),
    path('result', views.result, name='result'),
    path('upload', views.upload, name='upload'),
]