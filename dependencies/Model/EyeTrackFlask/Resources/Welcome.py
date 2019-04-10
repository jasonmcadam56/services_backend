from flask_restful import Resource
from flask import current_app as app


class Welcome(Resource):

    def get(self):
        statement = (
            'Hello user, this is the landing page for the model website '
        )
        return {'status': 'up', 'statement': statement}
