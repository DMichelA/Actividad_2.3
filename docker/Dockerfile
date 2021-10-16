FROM ubuntu:20.04

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get clean 

ENV requirements /home/requirements.txt

COPY requirements.txt ${requirements}

RUN pip3 install -r /home/requirements.txt 