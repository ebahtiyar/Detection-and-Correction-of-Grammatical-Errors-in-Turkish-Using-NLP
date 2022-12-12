from django.shortcuts import render

# Create your views here.
def register(request):
    return (request,"register.html")

def login(request):
    return(request,"login.html")

def logout(request):
    pass