# Using Sagemaker PyTorch container as base image
# from https://github.com/aws/sagemaker-pytorch-container
# DeepSpeed dependecies are installed inline with its container https://github.com/microsoft/DeepSpeed/blob/master/docker/Dockerfile

ARG REGION
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu110-ubuntu18.04
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}


##############################################################################
# Installation/Basic Utilities
##############################################################################

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common autotools-dev \
        nfs-common pdsh g++ gcc tmux less unzip \
        htop iftop iotop rsync iputils-ping \
        net-tools \
        sudo

# Install LLVM 3.9
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh 9

# VD: commeted out, SM container already exposes SSH port
##############################################################################
# Client Liveness & Uncomment Port 22 for SSH Daemon
##############################################################################
# # Keep SSH client alive from server side
# RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
# RUN cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
#     sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config


# VD: Mellanox OFED is driver for Infiniband and RoCE - should be irrelevant for AWS
# ##############################################################################
# # Mellanox OFED
# ##############################################################################
# ENV MLNX_OFED_VERSION=4.6-1.0.1.1
# RUN apt-get install -y libnuma-dev
# RUN cd ${STAGE_DIR} && \
#     wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64.tgz | tar xzf - && \
#     cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64 && \
#     ./mlnxofedinstall --user-space-only --without-fw-update --all -q && \
#     cd ${STAGE_DIR} && \
#     rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64*

# ##############################################################################
# # nv_peer_mem
# ##############################################################################
# ENV NV_PEER_MEM_VERSION=1.1
# ENV NV_PEER_MEM_TAG=1.1-0
# RUN mkdir -p ${STAGE_DIR} && \
#     git clone https://github.com/Mellanox/nv_peer_memory.git --branch ${NV_PEER_MEM_TAG} ${STAGE_DIR}/nv_peer_memory && \
#     cd ${STAGE_DIR}/nv_peer_memory && \
#     ./build_module.sh && \
#     cd ${STAGE_DIR} && \
#     tar xzf ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_VERSION}.orig.tar.gz && \
#     cd ${STAGE_DIR}/nvidia-peer-memory-${NV_PEER_MEM_VERSION} && \
#     apt-get update && \
#     apt-get install -y dkms && \
#     dpkg-buildpackage -us -uc && \
#     dpkg -i ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_TAG}_all.deb
    


##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3
# RUN apt-get install -y python3 python3-dev && \
#     rm -f /usr/bin/python && \
#     ln -s /usr/bin/python3 /usr/bin/python && \
#     curl -O https://bootstrap.pypa.io/get-pip.py && \
#         python get-pip.py && \
#         rm get-pip.py && \
#     pip install --upgrade pip && \
#     # Print python an pip version
#     python -V && pip -V
RUN pip install pyyaml
RUN pip install ipython

##############################################################################
# TensorFlow
##############################################################################
ENV TENSORFLOW_VERSION=1.15.2
RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION}


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile-dev \
        libcupti-dev \
        libjpeg-dev \
        libpng-dev \
        screen

RUN pip install yappi \
                cffi \
                ipdb \
                py3nvml \
                pyarrow \
                graphviz \
                astor \
                tqdm \
                sentencepiece \
                msgpack \
                sphinx \
                sphinx_rtd_theme \
                nvidia-ml-py3 \
                mpi4py \
                cupy-cuda101

##############################################################################
# PyYAML build issue
# https://stackoverflow.com/a/53926898
##############################################################################
RUN rm -rf /usr/lib/python3/dist-packages/yaml && \
    rm -rf /usr/lib/python3/dist-packages/PyYAML-*
    

##############################################################################
## Add deepspeed user
###############################################################################
# Add a deepspeed user with user id 8877
#RUN useradd --create-home --uid 8877 deepspeed
RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
RUN usermod -aG sudo deepspeed
RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
USER deepspeed


##############################################################################
# DeepSpeed
##############################################################################


# ENV CUDA_HOME=/usr/local/cuda-11.0
ENV CUDA_HOME=/usr/local/cuda-10.1

ADD https://raw.githubusercontent.com/aws/deep-learning-containers/master/src/deep_learning_container.py /usr/local/bin/deep_learning_container.py
RUN sudo chmod ugo+rwx /usr/local/bin/deep_learning_container.py

RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed

# copy scripts to directory under path
# ENV DEEPSPEED_BIN=/opt/ml/code/deepspeed/bin
# RUN sudo mkdir -p ${DEEPSPEED_BIN}
# RUN ls -la ${STAGE_DIR}/DeepSpeed/bin/
# RUN sudo cp -r ${STAGE_DIR}/DeepSpeed/bin/* ${DEEPSPEED_BIN}

# Placing DeepSpeed binaries on PATH
USER root
ENV PATH=/home/deepspeed/.local/bin:${PATH}
RUN sudo echo 'Defaults secure_path=\
    /opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin'\
    >> /etc/sudoers.d/deepspeed_config

USER deepspeed
RUN cd ${STAGE_DIR}/DeepSpeed && \
    git checkout . && \
    git checkout master && \
    ./install.sh -s
RUN rm -rf ${STAGE_DIR}/DeepSpeed
RUN python -c "import deepspeed; print(deepspeed.__version__)"

############# Configuring Sagemaker ##############
COPY training_container /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM launcher.py

USER root
WORKDIR /

