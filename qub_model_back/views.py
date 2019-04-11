import json
from json import loads

from celery.task.control import revoke
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from urllib3 import PoolManager

from backend_service import models
from qub_model_back.settings import EYETRACK_PROGRESS_DIR, DATASET_SAVE_LOCATION, EYETRACK_RESULTS_DIR
from qub_model_back.tasks import app
from backend_service.tasks import run

import os

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


def download_model(request, model_id):
    model = models.Model.objects.get(id=model_id)
    name = 'saved_model.pb'  # hardcoded as tensor always saves under the same name, this needs changed to model.name
    file_path = '{}/{}'.format(model.model_path, name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/force-download')
            response['Content-Disposition'] = 'inline; filename={}'.format(name)
            return response
    return HttpResponse(status=404)


@csrf_exempt
def upload_dataset(request):
    name = request.POST['name']
    dataset = request.FILES['file']

    file_path = DATASET_SAVE_LOCATION + name
    print(file_path)

    try:
        ds = models.DataSet.objects.create(name=name, file_path=file_path)

        with open(file_path, 'wb') as f:
            for chunk in dataset.chunks():
                f.write(chunk)

        ds.save()

        return HttpResponse(200)

    except Exception as e:
        return HttpResponseBadRequest(e)


@csrf_exempt
def test_model(request, model_id):

    if request.method == 'POST':
        try:
            dataset_id = loads(request.body.decode('utf-8')).get('dataset_id')

            model = models.Model.objects.get(id=model_id)
            dataset = models.DataSet.objects.get(id=dataset_id)

            _args = {
                'run_type': 'test',
                'nn_type': model.type,
                'nn_model': model.checkpoint_path,
                'dataset': dataset.file_path,
                'name': model.name
            }

            _kwargs = {
                'name': model.name,
                'task': 'test',
            }

            _args = parse_args(_args)
            print(_args)
            run.apply_async(args=_args)

            return HttpResponse(200)
        except Exception as e:
            print(e)
            return HttpResponse(e, status=500)

    elif request.method == 'GET':

        name = models.Model.objects.get(id=model_id).name
        file = EYETRACK_RESULTS_DIR + name + '.json'

        with open(file, 'r') as f:
            contents = f.read()

        contents = json.dumps(contents)
        return HttpResponse('{}'.format(contents), content_type='application/json')

    return HttpResponse(status=400)


def download_test_results(request, model_id):
    name = models.Model.objects.get(id=model_id).name
    file = EYETRACK_RESULTS_DIR + name + '.json'

    if os.path.exists(file):
        with open(file, 'r') as f:
            response = HttpResponse(f.read(), content_type='application/force-download')
            response['Content-Disposition'] = 'inline; filename={}'.format(name)
            return response

    return HttpResponse(status=404)


def revoke_task(request):

    if not request.method == 'POST':
        return HttpResponseBadRequest('Only http POST allowed for revoking tasks')
    print(request.POST)
    task_id = request.POST['task_id']
    revoke(task_id, terminate=True)
    return HttpResponse(200)


def append_progress(celery_dict, file_type='json'):

    if not celery_dict:
        return

    for host in celery_dict.keys():
        for worker in celery_dict[host]:
            meta = json.loads(worker['kwargs'].replace('\'', '"'))
            file_path = '{}/{}.{}'.format(EYETRACK_PROGRESS_DIR, meta['name'], file_type)

            try:
                with open(file_path, mode='r') as f:
                    worker['progress'] = json.loads(f.read())

            except FileNotFoundError as e:
                #  file wont be created straight away
                print('ERROR: FILE NOT FOUND: {}'.format(file_path))
                pass


def http_post(url, data, context):

    body = json.dumps(data)
    body = body.encode('utf-8')

    res = http.request('POST',
                       url,
                       body=body,
                       headers={'Content-Type': 'application/json'}
                       )

    context['STATUS_CODE'] = res.status


def parse_args(data):

    args = []

    if data.get('run_type') == 'test':
        args.append('--test')
        args.append('--type={}'.format(data.get('nn_type')))
        args.append('--modelLoc={}'.format(data.get('nn_model')))
        args.append('--data={}'.format(data.get('dataset')))
        args.append('--name={}'.format(data.get('name')))
    elif data.get('run_type') == 'train':
        args.append('--train')
        args.append('--type={}'.format(data.get('nn_type')))
        args.append('--data={}'.format(data.get('dataset_location')))
        args.append('-p={}'.format(data.get('name')))

    args.append('-v')

    return args
