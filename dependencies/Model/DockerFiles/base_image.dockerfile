FROM ubuntu:latest
# We are going to build with ubuntu

# First we need to install python, pip and create symlink for python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-distutils \
    curl &&  ln -s /usr/bin/python3 /usr/bin/python

# Store in temp folder for pip install
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
RUN python /tmp/get-pip.py
# TODO: Make python pip package of our software install it instead.
RUN pip install --user --upgrade tensorflow-gpu
RUN pip install numpy
# Local data should be stored in the data folder then add to the docker image. 
# This allows us to rebuild the system faster when making changes
# Also will allow support for when we don't need data. E.g. Already trained model.
ADD data/* /usr/src/ 

