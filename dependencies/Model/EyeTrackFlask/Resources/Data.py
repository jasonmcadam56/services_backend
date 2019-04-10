import os
from flask_restful import Resource
from flask import current_app as app
from flask import abort


class Data(Resource):

    """
        This class is used to represent our data folders so that the user can see
        what they have available to them.

        Supported API calls: GET (List)
    """

    def get(self, **kwargs):
        """
            Return all the files in the DATA_LOC directory by check if they are label
            with our arch types.
        """
        path = app.config['DATA_LOC']
        files = os.listdir(path)
        return_value = {'File': []}
        for f in files:
            sp_f = f.split('_')
            if len(sp_f) < 2:
                continue
            if sp_f[1] not in ['mit', 'eyeq', 'npza']:
                continue
            return_value['File'].append({'File Name': f, 'Arch Type': sp_f[1]})
        return return_value

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
