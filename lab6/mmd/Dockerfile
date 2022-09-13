FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install \
    mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html \
    mmengine==0.1.0 \
    shapely==1.8.4 \
    tensorboard==2.10.0

RUN git clone https://github.com/open-mmlab/mmdetection.git
WORKDIR mmdetection
RUN git checkout tags/v2.25.1 \
    && pip install -v -e . \
    && echo "data/" >> .gitignore && echo "work_dirs/" >> .gitignore
    