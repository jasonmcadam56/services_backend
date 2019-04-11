import json

from EyeTrack import runner

from backend_service.models import Model
from qub_model_back.settings import EYETRACK_PROGRESS_DIR
from qub_model_back.tasks import app


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


    print(args)

    runner.main(args)

    if '--train' in args:
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

@app.task
def debug(*args, **kwargs):
    """
    :param args: any args
    :param kwargs: amy kwargs
    :return: nothing

    A simple method used to check the input of a message and that celery will execute it async

    """
    print('Debug called with:\n{}\n{}'.format(args, kwargs))
