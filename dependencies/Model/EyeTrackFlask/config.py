import os

basedir = os.path.abspath(os.path.dirname(__file__))

DEBUG = os.environ.get('DEBUG', default=False)
MODEL_LOC = os.getenv('MODEL_FOLDER', '{}{}{}'.format(os.path.abspath(os.path.dirname(__file__)), os.sep, 'eyeq'))
DATA_LOC = os.getenv('DATA_FOLDER', '{}{}{}'.format(os.path.abspath(os.path.dirname(__file__)), os.sep, 'data'))
PROGRESS_LOC = os.getenv('PROGRESS_FOLDER', '{}{}{}'.format(os.path.abspath(os.path.dirname(__file__)), os.sep, 'progress'))
TEST_LOC = os.getenv('TEST_FOLDER', '{}{}{}'.format(os.path.abspath(os.path.dirname(__file__)), os.sep, 'test_data'))
