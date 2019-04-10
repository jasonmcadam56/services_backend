from flask_restful import Resource, reqparse
from flask import current_app as app
from flask import abort
from EyeTrack import runner


class Train(Resource):

    """
        This class is used to represent our data folders so that the user can see
        what they have available to them.

        Supported API calls: GET (List), POST (Create).
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('train')
        self.parser.add_argument('dataArch')
        self.parser.add_argument('modelLoc')
        self.parser.add_argument('retrain')
        self.parser.add_argument('data')
        self.parser.add_argument('v')
        self.parser.add_argument('epoch')
        self.parser.add_argument('batch_size')
        self.parser.add_argument('type')
        self.parser.add_argument('screenSizeX')
        self.parser.add_argument('screenSizeY')
        self.parser.add_argument('gridRowSize')
        self.parser.add_argument('gridColSize')
        self.parser.add_argument('progressFile')
        super().__init__()

    def get(self):
        """
            Returns the flag information for training a model.
        """
        return {
            'Flags': {
                'train': 'tell us to train the model.',
                'dataArch': 'What type of data format are you using MIT, EyeQ.',
                'modelLoc': 'Model that you wish to use (Needed for retrain).',
                'retrain': 'Allows for the retraining of the model.',
                'data': 'Data you want the model to be trained on',
                'type': 'What model type CNN, GRID etc',
                'v': 'Verbose',
                'epoch': 'number of epochs',
                'batch_size': 'size of the batches',
                'progressFile': 'the name of the progress file you wish to save',
                'screenSizeX': 'the horizontal size of the screen',
                'screenSizeY': 'the vertical size of the screen',
                'gridRowSize': 'the size of the row per grid',
                'gridColSize': 'the size of the column per grid'

            },
            'How to use': (
                'Post a array of flags with the correct settings e.g.'
                ' "flags":{ '
                ' "train": true, '
                ' "data": "10_eyeq_dataset_split.npz", '
                ' "type": "cnn", '
                ' "-v": true '
            )
        }

    def post(self):
        """
            A pain in the arse
        """
        kwargs = self.parser.parse_args()

        args = ['--train']
        path = app.config['DATA_LOC']

        for i in ['type', 'data']:
            if not kwargs.get(i):
                raise ValueError('Missing {} flag'.format(i))

        args.append('--data={}'.format(path + '\\' + kwargs['data']))
        args.append('--type={}'.format(kwargs.get('type')))

        if kwargs.get('v'):
            args.append('-v')

        if kwargs.get('retrain') and kwargs.get('modelLoc'):
            args.append('--retrain')
            args.append('--modelLoc={}'.format(app.config['MODEL_LOC'] + '\\' + kwargs['modelLoc']))
        elif kwargs.get('retrain') and not kwargs.get('modelLoc'):
            raise ValueError('Missing modelLoc kwargs')
        elif not kwargs.get('retrain') and kwargs.get('modelLoc'):
            raise ValueError('Missing retrain kwargs')

        for i in ['dataArch', 'epoch', 'batch_size', 'screenSizeX', 'screenSizeY', 'gridRowSize', 'gridColSize', 'progressFile']:
            if kwargs.get(i):
                args.append('--{}={}'.format(i, kwargs.get(i)))
        print(args)
        return runner.main(from_app=args)

    def put(self):
        """
            Not supported for Data.
        """
        return abort(405)
