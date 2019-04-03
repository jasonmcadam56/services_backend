import json

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest
from urllib3 import PoolManager

from backend_service import models

from qub_model_back.tasks import app

from django.views.decorators.csrf import csrf_exempt
from celery.task.control import revoke

from json import loads

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
        content = json.dumps(content)

    elif worker_id and request.method == 'GET':
        #/workers/<uuid>/
        content = app.control.inspect().active()
        host_name = list(content.keys())[0]
        content = content[host_name]

        for worker in content:
            if worker['id'] == worker_id:
                content = worker
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



def http_post(url, data, context):

    body = json.dumps(data)
    body = body.encode('utf-8')

    res = http.request('POST',
                       url,
                       body=body,
                       headers={'Content-Type': 'application/json'}
                       )

    context['STATUS_CODE'] = res.status

