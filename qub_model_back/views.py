from django.shortcuts import redirect, render
from django.conf import settings
from django.utils.safestring import mark_safe # used to fire off json
import json
from urllib3 import PoolManager

http = PoolManager()


def index(request):

    context = {}

    if request.method == 'POST':
        res = http.request('POST', settings.FRONT_END_URL, fields={
            'message': request.POST.get('message')
        })
        context['STATUS_CODE'] = res.status

    return render(request, 'qub_model_back/index.html', context)
