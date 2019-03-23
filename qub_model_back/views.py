import json
import os

from django.conf import settings
from django.shortcuts import render
from urllib3 import PoolManager

from qub_model_back.tasks import run

http = PoolManager()


def index(request):

    context = {
        'datasets': settings.DATASETS,
        'nn_models': settings.NNMODELS,
        'nn_types': settings.NNTYPES
    }

    if request.method == 'POST':
        args = parse_args(request.POST)
        print(args)
        run(args)

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


def parse_args(data):

    args = []

    if data.get('run_type') == 'test':
        args.append('--test')
        args.append('--modelLoc=={}'.format(os.path.abspath('qub_model_back/nnmodels/{}'.format(data.get('nn_model')))))
        args.append('--data={}'.format(os.path.abspath('qub_model_back/datasets/{}'.format(data.get('dataset')))))
        args.append('--type={}'.format(data.get('nn_type')))
    else:  # train
        pass
        # args.append('--type={}'.format(data.get('nn_type')))

    args.append('-v')

    return args
