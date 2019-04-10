import os

from flask import Flask, jsonify, make_response
from flask_restful import Resource, Api

from Resources.Welcome import Welcome
from Resources.Trainer import Train
from Resources.Tester import Test
from Resources.Data import Data
from Resources.ML_Model import ML_Model
from Resources.Progress import Progress, ProgressList


def create_app(config_filename):
    """
        Setup of the EyeTrackFlask Application.

        Raises:
            ValueError: If the Progress, data or Model folders are missing or not setup.
    """
    app = Flask(__name__)
    api = Api(app)
    app.config.from_pyfile(config_filename)
    err_msg = []

    for i in [app.config['PROGRESS_LOC'], app.config['DATA_LOC'], app.config['MODEL_LOC']]:
        if not os.path.exists(i):
            err_msg.append(i)

    if err_msg:
        raise ValueError('The following folder are missing {}, Make sure these folder exists or setup the right config.'.format(err_msg))

    api.add_resource(Welcome, '/')
    api.add_resource(Train, '/train')
    api.add_resource(Data, '/data')
    api.add_resource(Test, '/test')
    api.add_resource(ML_Model, '/model')
    api.add_resource(ProgressList, '/progress')
    api.add_resource(Progress, '/progress/<string:progress_id>')

    return app


if __name__ == '__main__':
    """
        Runs the application for us.
    """
    app = create_app('config.py')
    app.run(debug=True)
