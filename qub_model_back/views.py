from django.shortcuts import render
from django.utils.safestring import mark_safe # used to fire off json
import json


def index(request):
    return render(request, 'qub_model_back/index.html', {})
