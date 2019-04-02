import json

from django.shortcuts import render
from django.http import HttpResponse
from urllib3 import PoolManager

from backend_service import models

from qub_model_back.celery import app

http = PoolManager()


def index(request):

    context = {
        'nn_types': [t for t in models.Model.TYPE_CHOICES],
        'datasets': [dataset for dataset in models.DataSet.objects.all()]
    }

    return render(request, 'qub_model_back/index.html', context)


def worker(request):

    active = app.control.inspect().active()  # returns active nodes
    # get progress status here and append the active dir

    return HttpResponse('{}'.format(active), content_type='application/json')


def http_post(url, data, context):

    body = json.dumps(data)
    body = body.encode('utf-8')

    res = http.request('POST',
                       url,
                       body=body,
                       headers={'Content-Type': 'application/json'}
                       )

    context['STATUS_CODE'] = res.status

