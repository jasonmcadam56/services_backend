from flask_restful import Resource, reqparse
from flask import current_app as app
from flask import abort
from EyeTrack import runner
from pathlib import Path

import os
import json


class Test(Resource):

    """
        This class is used to represent our data folders so that the user can see
        what they have available to them.

        Supported API calls: GET (List), POST (Create).
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('test')
        self.parser.add_argument('name')
        self.parser.add_argument('dataArch')
        self.parser.add_argument('modelLoc')
        self.parser.add_argument('data')
        self.parser.add_argument('v')
        self.parser.add_argument('epoch')
        self.parser.add_argument('batch_size')
        self.parser.add_argument('type')
        self.parser.add_argument('screenSizeX')
        self.parser.add_argument('screenSizeX')
        self.parser.add_argument('gridRowSize')
        self.parser.add_argument('gridColSize')

        super().__init__()

    def get(self):
        """
            Returns all files in TEST_LOC dir.

            Returns: 
                (json) All the files in TEST_LOC
        """
        path = app.config['TEST_LOC']
        files = os.listdir(path)
        return_value = {'File': []}
        for f in files:
            return_value['File'].append({'Name': f})
        return return_value

    def post(self):
        """
            Create the test flags for passing to the model.

            Returns: 
                (json) The test data of the model.
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

        for i in ['dataArch', 'epoch', 'batch_size', 'screenSizeX', 'screenSizeY', 'gridRowSize', 'gridColSize', 'name']:
            if kwargs.get(i):
                args.append('--{}={}'.format(i, kwargs.get(i)))

        return runner.main(from_app=args)

    def put(self):
        """
            Not supported for Data.
        """
        return abort(405)


class TestFile(Resource):

    """
        This class is used to represent TestFiles so that users can access the file 
        later.

        Supported API calls: GET (List)
    """

    def get(self, test_file, **kwargs):
        """
            Return a file in the TEST_LOC directory and read the file content.

            Raises:
                404 (HTTP Status code) if the file is missing.
            Returns: 
                (json) The file content.
        """
        path = app.config['TEST_LOC']
        l_file = Path('{}/{}'.format(path, test_file))
        if not l_file.is_file():
            abort(404)

        data = {'file_name': test_file}

        with open(l_file) as j_file:
            data.update(json.load(j_file))

        return data

    def post(self):
        """
            Not supported for Data.
        """
        return abort(405)

    def put(self):
        """
            Not supported for Data.
        """
        return abort(405)
