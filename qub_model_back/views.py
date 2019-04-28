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
from backend_service.tasks import run, run_prediction

import cv2 as cv
import numpy as np
import base64
import json


import os

http = PoolManager()

def index(request):

    """
    :param request: django request object
    :return: rendered html of index template
    """

    context = {
        'nn_types': [t for t in models.Model.TYPE_CHOICES],
        'datasets': [dataset for dataset in models.DataSet.objects.all()]
    }

    return render(request, 'qub_model_back/index.html', context)


@csrf_exempt
def worker(request, worker_id=None):
    """
    :param request: django request object
    :param worker_id: id for worker to interact with
    :return: content of interaction
    :except: exception on bad data
    """
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


@csrf_exempt
def predict(request, model_id):
    """
    :param request: django request object
    :return: model predictions
    """

    if request.method != 'POST' or not model_id:
        return HttpResponse(400)

    model = models.Model.objects.get(id=model_id)
    ds_save_loc = '/home/jason/Desktop/sample_dataset.npz'

    image_data = json.loads(request.body.decode('utf-8'))

    image_data['face_mask'] = cv.imdecode(np.fromstring(base64.b64decode(image_data['face_mask']), np.uint8),
                                        cv.IMREAD_UNCHANGED)

    output = dict()

    for key, value in image_data.items():
        if not ('rect' in key or 'mask' in key):
            try:
                rawImage = base64.b64decode(value)
                image = cv.imdecode(np.fromstring(rawImage, np.uint8), cv.IMREAD_UNCHANGED)
                output['{}_{}'.format('train', key)] = np.zeros((0, 0))
                output['{}_{}'.format('val', key)] = np.array([image], np.uint8)
            except:
                print(key)

    faceMask = np.zeros((25, 25), np.uint8)

    for x in range(0, 25):
        for y in range(0, 25):
            faceMask[x][y] = round(int(image_data['face_mask'][x][y][0]) / 255.0)

    output['val_face_mask'] = [faceMask]
    output['train_face_mask'] = np.zeros((0, 0))
    np.savez_compressed(ds_save_loc, **output)

    if model.type == models.Model.CNN:
        model_type = 'cnn'
    elif model.type == models.Model.GCNN:
        model_type = 'gcnn'

    _args = parse_args({
        'nn_type': model_type,
        'run_type': 'test',
        'nn_model': model.model_path,
        'dataset_location': ds_save_loc,
        'name': model.name
    })

    _args.append('-dl')

    _kwargs = {
        'name': model.name,
        'callback': 'http://localhost:5000/heatmap/'
    }

    run_prediction.apply_async(args=_args, kwargs=_kwargs)
    return HttpResponse(200)


def download_model(request, model_id):
    """
    :param request: django request object
    :param model_id: ID of model to download
    :return: stream for client to download model file
    """
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
    """
    :param request: django request object
    :return: 200 on successfull upload
    :except: return Exception data on expection
    """
    name = request.POST['name']
    dataset = request.FILES['file']

    file_path = DATASET_SAVE_LOCATION + name

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
    """
    :param request: django request object
    :param model_id: ID of model to use for testing/valiation
    :return: HTTP 200
    :except Dataset or Model not found. Test files results inaccessible
    """
    model = models.Model.objects.get(id=model_id)

    if request.method == 'POST':
        try:
            dataset_id = loads(request.body.decode('utf-8')).get('dataset_id')

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

            run.apply_async(args=_args)

            return HttpResponse(200)
        except Exception as e:
            return HttpResponse(e, status=500)

    elif request.method == 'GET':

        name = models.Model.objects.get(id=model_id).name

        if model.type == models.Model.CNN:
            file = '{}/{}.json'.format(EYETRACK_RESULTS_DIR, name)
        elif model.type == models.Model.GCNN:
            file = '{}/gcnn_test_{}.json'.format(EYETRACK_RESULTS_DIR, name)

        with open(file, 'r') as f:
            contents = f.read()

        contents = json.dumps(contents)
        return HttpResponse('{}'.format(contents), content_type='application/json')

    return HttpResponse(status=400)


def download_test_results(request, model_id):
    """
    :param request: django request object
    :param model_id: ID of model to download
    :return: Stream to download model
    :except 404 when test file isnt found
    """
    name = models.Model.objects.get(id=model_id).name
    file = EYETRACK_RESULTS_DIR + name + '.json'

    if os.path.exists(file):
        with open(file, 'r') as f:
            response = HttpResponse(f.read(), content_type='application/force-download')
            response['Content-Disposition'] = 'inline; filename={}'.format(name)
            return response

    return HttpResponse(status=404)


def revoke_task(request):
    """
    :param request: django request object
    :return: HTTP response 200
    """

    if not request.method == 'POST':
        return HttpResponseBadRequest('Only http POST allowed for revoking tasks')

    task_id = request.POST['task_id']
    revoke(task_id, terminate=True)
    return HttpResponse(200)


def append_progress(celery_dict, file_type='json'):
    """
    :param celery_dict: dict returned from celery.active method
    :param file_type: type of file to read
    :return: dict of original data with worker progress appended
    """

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
    """
    :param url: host name to post to
    :param data: data of post request
    :param context: django context to append response data to
    :return:
    """

    body = json.dumps(data)
    body = body.encode('utf-8')

    res = http.request('POST',
                       url,
                       body=body,
                       headers={'Content-Type': 'application/json'}
                       )

    context['STATUS_CODE'] = res.status


def parse_args(data):
    """
    :param data: dict of data for passing to EyeTrack runner
    :return: data formatted to be accessible by EyeTrack runner
    """

    args = []

    if data.get('run_type') == 'test':
        args.append('--test')
        args.append('--type={}'.format(data.get('nn_type')))
        args.append('--modelLoc={}'.format(data.get('nn_model')))
        args.append('--data={}'.format(data.get('dataset_location')))
        args.append('--name={}'.format(data.get('name')))
    elif data.get('run_type') == 'train':
        args.append('--train')
        args.append('--type={}'.format(data.get('nn_type')))
        args.append('--data={}'.format(data.get('dataset_location')))
        args.append('-p={}'.format(data.get('name')))
        if data.get('max_epoch'):
            args.append('-e={}'.format(data.get('max_epoch')))

    args.append('-v')

    return args
