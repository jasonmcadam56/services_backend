from django.shortcuts import redirect, render
from django.conf import settings
from django.utils.safestring import mark_safe # used to fire off json
import json
from urllib3 import PoolManager

http = PoolManager()


def index(request):

    context = {}

    if request.method == 'POST':
        http_post(settings.FRONT_END_URL, {'message': request.POST['message']}, context)

    return render(request, 'qub_model_back/index.html', context)


def http_post(url, data, context):

    body = json.dumps(data)
    body = body.encode('utf-8')

    res = http.request('POST',
                       url,
                       body=body,
                       headers={'Content-Type': 'application/json'}
                       )

    context['STATUS_CODE'] = res.status
