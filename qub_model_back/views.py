import json
from json import loads

from celery.task.control import revoke
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from urllib3 import PoolManager

from backend_service import models
from qub_model_back.settings import EYETRACK_PROGRESS_DIR
from qub_model_back.tasks import app

http = PoolManager()

def index(request):

    context = {
        'nn_types': [t for t in models.Model.TYPE_CHOICES],
        'datasets': [dataset for dataset in models.DataSet.objects.all()]
    }

    return render(request, 'qub_model_back/index.html', context)


@csrf_exempt
def worker(request, worker_id=None):

    content = ''

    if worker_id == None:
        #/workers/
        content = app.control.inspect().active()
        append_progress(content)
        content = json.dumps(content)

    elif worker_id and request.method == 'GET':
        #/workers/<uuid>/
        content = app.control.inspect().active()
        host_name = list(content.keys())[0]
        content = content[host_name]

        for worker in content:
            if worker['id'] == worker_id:
                content = worker
                append_progress(content)
                content = json.dumps(content)
                break

    elif worker_id and request.method == 'POST':
        action = loads(request.body.decode('utf-8')).get('action')

        if action == None:
            return HttpResponseBadRequest()

        if action == 'revoke':
            revoke(worker_id, terminate=True)

    return HttpResponse('{}'.format(content), content_type='application/json')


def revoke_task(request):

    if not request.method == 'POST':
        return HttpResponseBadRequest('Only http POST allowed for revoking tasks')

    task_id = request.POST['task_id']
    revoke(task_id, terminate=True)
    return HttpResponse(200)


def append_progress(celery_dict, file_type='json'):

    for host in celery_dict.keys():
        for worker in celery_dict[host]:
            meta = json.loads(worker['kwargs'].replace('\'', '"'))
            file_path = '{}/{}.{}'.format(EYETRACK_PROGRESS_DIR, meta['name'], file_type)

            with open(file_path, mode='r') as f:
                worker['progress'] = json.loads(f.read())


def http_post(url, data, context):

    body = json.dumps(data)
    body = body.encode('utf-8')

    res = http.request('POST',
                       url,
                       body=body,
                       headers={'Content-Type': 'application/json'}
                       )

    context['STATUS_CODE'] = res.status
