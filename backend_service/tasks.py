import json

from EyeTrack import runner

from backend_service.models import Model
from qub_model_back.settings import EYETRACK_PROGRESS_DIR, EYETRACK_RESULTS_DIR
from qub_model_back.tasks import app
from urllib3 import PoolManager


http = PoolManager()

@app.task
def run(*args, **kwargs):
    """
    :param args (list):
        --type (str)    : model type ie cnn or grid         eg... '--type=cnn'
        --test (flag)   : pass in to only do a test run     eg... '--test'
        --modelLoc (str): path to the model to be used      eg... '--modelLoc=<path>'
        --data(str)     : data set to pass to the model     eg... '--data=<path>'

        eg...   args = ['--type=cnn', 'test', ...]
    """

    # --type = gcnn
    runner.main(args)

    # train regression cnn
    if '--train' in args and '--type=cnn' in args:
        file_path = '{}/{}.json'.format(EYETRACK_PROGRESS_DIR, kwargs['name'])

        try:
            with open(file_path, 'r') as f:
                data = f.read()
                data = json.loads(data)

                model = Model.objects.get(name=kwargs['name'])

                model.model_path = data['model_simple_loc']
                model.checkpoint_path = data['checkpoints'] + data['model_name']
                model.status = Model.COMPLETE
                model.save()
        except FileNotFoundError as e:
            print('ERROR: File not found:'.format(file_path))

    # train grided cnn


@app.task
def run_prediction(*args, **kwargs):

    print(args)

    runner.main(args)

    if '--type=cnn' in args:
        file_path = '{}/{}.json'.format(EYETRACK_RESULTS_DIR, kwargs['name'])
    if '--type=gcnn' in args:
        file_path = '{}/gcnn_test_{}.json'.format(EYETRACK_RESULTS_DIR, kwargs['name'])


    try:
        with open(file_path, 'r') as f:
            contents = f.read()
            if '--type=cnn' in args:
                contents = json.loads(contents)
            if '--type=gcnn' in args:
                contents = contents.replace('\'', '"')[1:-1]
                contents = json.loads(contents)


    except FileNotFoundError as e:
        print(e)

    if '--type=cnn' in args:
        point = contents['pred']['model'][0]
        stats = contents['stats']
        body = {
            "point": {
                "x": abs(point[0]),
                "y": abs(point[1])
            },
            "data": stats
        }

        body = json.dumps(body)

        http.request('POST',
                     'http://localhost:5000/heatmap/',
                     body=body)

    if '--type=gcnn' in args:

        sector = contents['pred']['model']
        stats = contents['stats']

        body = {
            "sector": sector,
            "data": stats
        }

        body = json.dumps(body)

        http.request('POST',
                     'http://localhost:5000/heatmap/',
                     body=body)

@app.task
def debug(*args, **kwargs):
    """
    :param args: any args
    :param kwargs: amy kwargs
    :return: nothing

    A simple method used to check the input of a message and that celery will execute it async

    """
    print('Debug called with:\n{}\n{}'.format(args, kwargs))
