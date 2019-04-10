import os
import json

from pathlib import Path
from flask_restful import Resource
from flask import current_app as app
from flask import abort


class ProgressList(Resource):

    """
        This class is used to represent our Progress folders so that the user can see
        what they have available to them.

        Supported API calls: GET (List)
    """

    def get(self, **kwargs):
        """
            Return all the files in the PROGRESS_LOC directory.
        """
        path = app.config['PROGRESS_LOC']
        files = os.listdir(path)
        return_value = {'File': []}
        for f in files:
            return_value['File'].append({'Name': f})
        return return_value

    def post(self):
        """
            Not supported for ProgressList.
        """
        return abort(405)

    def put(self):
        """
            Not supported for ProgressList.
        """
        return abort(405)


class Progress(Resource):

    """
        This object is used to read a progress file and return it's data
        for the user.

        Supports API Call: GET (Object)
    """

    def get(self, progress_id, **kwargs):
        """
            Return a file in the PROGRESS_LOC directory and read the file content.

            Raises:
                404 (HTTP Status code) if the file is missing.
        """
        path = app.config['PROGRESS_LOC']
        l_file = Path('{}/{}'.format(path, progress_id))
        if not l_file.is_file():
            abort(404)

        data = {'file_name': progress_id}

        with open(l_file) as j_file:
            data.update(json.load(j_file))

        return data

    def post(self):
        """
            Not supported for Progress.
        """
        return abort(405)

    def put(self):
        """
            Not supported for Progress.
        """
        return abort(405)
