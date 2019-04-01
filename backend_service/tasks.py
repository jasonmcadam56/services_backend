from qub_model_back.celery import app
from EyeTrack import runner


@app.task
def run(*args):
    """
    :param args (list):
        --type (str)    : model type ie cnn or grid         eg... '--type=cnn'
        --test (flag)   : pass in to only do a test run     eg... '--test'
        --modelLoc (str): path to the model to be used      eg... '--modelLoc=<path>'
        --data(str)     : data set to pass to the model     eg... '--data=<path>'

        eg...   args = ['--type=cnn', 'test', ...]
    """

    runner.main(args)

@app.task
def debug(*args, **kwargs):
    """
    :param args: any args
    :param kwargs: amy kwargs
    :return: nothing

    A simple method used to check the input of a message and that celery will execute it async

    """
    print('Debug called with:\n{}\n{}'.format(args, kwargs))
