from django.contrib import admin
from django.urls import path
from .import views


app_name = "correction"

urlpatterns = [
    path("norm/",views.normalization,name = "normalization"),
    path("ans/",views.answer,name = "answer")
]


