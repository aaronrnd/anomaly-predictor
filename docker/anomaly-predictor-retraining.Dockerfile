FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04 as builder

ARG CONDA_ENV_FILE="anomaly-predictor-conda-env.yml"
ARG CONDA_ENV_NAME="anomaly-predictor"
ARG PROJECT_USER="abb"
ARG HOME_DIR="/home/$PROJECT_USER"

ARG DVC_VERSION="2.8.3"
ARG DVC_BINARY_NAME="dvc_2.8.3_amd64.deb"

ARG CONDA_HOME="$HOME_DIR/miniconda3"
ARG CONDA_BIN="$CONDA_HOME/bin/conda"
ARG MINI_CONDA_SH="Miniconda3-latest-Linux-x86_64.sh"

ENV PATH $CONDA_HOME/bin:$HOME_DIR/.local/bin:$PATH
WORKDIR $HOME_DIR

RUN groupadd -g 2222 $PROJECT_USER && useradd -u 2222 -g 2222 -m $PROJECT_USER

RUN touch "$HOME_DIR/.bashrc"

RUN apt-get update && \
    apt-get install --no-install-recommends -y bzip2 curl wget gcc rsync git vim locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://repo.anaconda.com/miniconda/$MINI_CONDA_SH && \
    chmod +x $MINI_CONDA_SH && \
    ./$MINI_CONDA_SH -b -p $CONDA_HOME && \
    rm $MINI_CONDA_SH

COPY --chown=2222:2222 $CONDA_ENV_FILE .

RUN $CONDA_BIN env create -f $CONDA_ENV_FILE && \
    $CONDA_BIN init bash 

RUN $CONDA_BIN install -c conda-forge conda-pack

RUN $CONDA_BIN pack -n $CONDA_ENV_NAME -o /tmp/env.tar && \
    mkdir /$CONDA_ENV_NAME && cd /$CONDA_ENV_NAME && tar xf /tmp/env.tar && \
    rm /tmp/env.tar

RUN /$CONDA_ENV_NAME/bin/conda-unpack

FROM nvidia/cuda:11.3.0-base-ubuntu18.04

ARG CONDA_ENV_NAME="anomaly-predictor"
ARG PROJECT_USER="abb"
ARG HOME_DIR="/home/$PROJECT_USER"

RUN groupadd -g 2222 $PROJECT_USER && useradd -u 2222 -g 2222 -m $PROJECT_USER
RUN echo "source /$CONDA_ENV_NAME/bin/activate" >>   "$HOME_DIR/.bashrc"

WORKDIR $HOME_DIR
RUN mkdir data models conf 

COPY --chown=2222:2222 scripts/retraining.sh scripts/
COPY --chown=2222:2222 src/anomaly_predictor src/anomaly_predictor
RUN touch src/__init__.py 

RUN chown 2222:2222 $HOME_DIR && \
    chown 2222:2222 $HOME_DIR/.bashrc && \
    chown 2222:2222 $HOME_DIR/data && \
    chown 2222:2222 $HOME_DIR/models && \
    chown 2222:2222 $HOME_DIR/conf && \
    rm /bin/sh && ln -s /bin/bash /bin/sh

COPY --from=builder /$CONDA_ENV_NAME /$CONDA_ENV_NAME

ENV CONDA_DEFAULT_ENV $CONDA_ENV_NAME
ENV PYTHONIOENCODING utf8
ENV LANG "C.UTF-8"
ENV LC_ALL "C.UTF-8"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

USER 2222
EXPOSE 8080

CMD ["sh", "scripts/retraining.sh"]