from flask_restful import Resource, reqparse
from flask import current_app as app
from flask import abort
from EyeTrack import runner


class Test(Resource):

    """
        This class is used to represent our data folders so that the user can see
        what they have available to them.

        Supported API calls: GET (List), POST (Create).
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('test')
        self.parser.add_argument('dataArch')
        self.parser.add_argument('modelLoc')
        self.parser.add_argument('data')
        self.parser.add_argument('v')
        self.parser.add_argument('epoch')
        self.parser.add_argument('batch_size')
        self.parser.add_argument('type')
        self.parser.add_argument('--screenSizeX')
        self.parser.add_argument('--screenSizeX')
        self.parser.add_argument('--gridRowSize')
        self.parser.add_argument('--gridColSize')

        super().__init__()

    def get(self):
        """
            Returns the flag information for testing a model.
        """
        return {
            'Flags': {
                'test': 'tell us to test the model.',
                'dataArch': 'What type of data format are you using MIT, EyeQ.',
                'modelLoc': 'Model that you wish to use (Needed for retrain).',
                'data': 'Data you want the model to be trained on',
                'type': 'What model type CNN or GCNN',
                'v': 'Verbose',
                'epoch': 'number of epochs',
                'batch_size': 'size of the batches',
                'screenSizeX': 'the horizontal size of the screen',
                'screenSizeY': 'the vertical size of the screen',
                'gridRowSize': 'the size of the row per grid',
                'gridColSize': 'the size of the column per grid'
            },
            'How to use': (
                'Post a json of flags with the correct settings e.g.'
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

        args = ['--test']
        path = app.config['DATA_LOC']

        for i in ['type', 'data', 'modelLoc']:
            if not kwargs.get(i):
                raise ValueError('Missing {} flag'.format(i))

        args.append('--data={}'.format(path + '\\' + kwargs['data']))
        args.append('--type={}'.format(kwargs.get('type')))
        args.append('--modelLoc={}'.format(app.config['MODEL_LOC'] + '\\' + kwargs['modelLoc'].split('.')[0]))

        if kwargs.get('v'):
            args.append('-v')

        for i in ['dataArch', 'epoch', 'batch_size', 'screenSizeX', 'screenSizeY', 'gridRowSize', 'gridColSize']:
            if kwargs.get(i):
                args.append('--{}={}'.format(i, kwargs.get(i)))

        return runner.main(from_app=args)

    def put(self):
        """
            Not supported for Data.
        """
        return abort(405)
