FROM model:latest

RUN mkdir /usr/src/eyeq
ADD ../EyeTrack/*.py /usr/src/eyeq/EyeTrack
ADD ../setup.py /usr/src/eyeq
ADD requirements.txt /usr/src/eyeq

RUN pip install -e /usr/src/eyeq
RUN echo 'alias debug_eye="python /usr/src/eyeq/runner.py -d /usr/src/eye_tracker_train_and_val.npz --type cnn --train -v"' >> ~/.bashrc