FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt -y install ssh
RUN apt -y install libopenmpi-dev
RUN apt -y install cmake
#RUN pip3 install tensorflow_addons
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install transformers
RUN pip3 install accelerate
