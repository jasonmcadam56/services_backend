from setuptools import setup
from setuptools import find_packages

setup(
    name='EyeTrack',
    version='1.0.0.',
    description='Eye tracking project from Lectrefy Inc.',
    author='Lectrefy Inc.',
    packages=find_packages(),
    install_requirements=[
        'absl-py==0.7.0',
        'astor==0.7.1',
        'dlib==19.16.0',
        'gast==0.2.2',
        'grpcio==1.18.0',
        'h5py==2.9.0',
        'imutils==0.5.2',
        'Keras-Applications==1.0.6',
        'Keras-Preprocessing==1.0.5',
        'Markdown==3.0.1',
        'numpy==1.16.0',
        'opencv-python==4.0.0.21',
        'pkg-resources==0.0.0',
        'protobuf==3.6.1',
        'six==1.12.0',
        'tensorboard==1.12.2',
        'tensorflow-gpu==1.12.0',
        'termcolor==1.1.0',
        'Werkzeug==0.14.1',
        'tox']
)
