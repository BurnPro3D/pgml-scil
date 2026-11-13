FROM quay.io/jupyter/base-notebook:latest

WORKDIR /home/jovyan/work

USER root

RUN apt-get update && apt-get install -y \
    git

RUN chmod a+x /usr/bin/git

RUN chown -R jovyan /home/jovyan

# make sudo access without password
RUN echo 'jovyan ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Install the requests library
RUN pip install requests

# install git extension
RUN pip install --upgrade jupyterlab==4.2.4


USER jovyan

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt