import os
from flask_restful import Resource
from flask import current_app as app
from flask import abort


class ML_Model(Resource):

    """
        This class is used to represent our Model folder so that the user can see
        what they have available to them.

        Supported API calls: GET (List)
    """

    def get(self, **kwargs):
        """
            Return all the files in the MODEL_LOC directory by check if they are label
            with our arch types.
        """
        path = app.config['MODEL_LOC']
        files = os.listdir(path)
        return_value = {'Models': []}
        already_added_filenames = []

        for f in files:
            sp_f = f.split('_')
            sp_n = f.split('.')
            sp_check = f.split('-')
            if len(sp_f) < 2 or len(sp_check) < 2:
                continue
            if sp_n[0] not in already_added_filenames:
                already_added_filenames.append(sp_n[0])
                return_value['Models'].append({'Model Name': sp_n[0], 'Model Type': sp_f[0]})

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
