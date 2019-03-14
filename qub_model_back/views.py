from django.shortcuts import redirect, render
from django.conf import settings
from django.utils.safestring import mark_safe # used to fire off json
import json


def index(request):
    return render(request, 'qub_model_back/index.html', {
        'FRONT_END_WS_URL': settings.FRONT_END_WS_URL
    })


def message(request):
    print(request.POST.get('message'))
    return redirect(index)
