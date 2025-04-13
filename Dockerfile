FROM ubuntu:22.04

#make non-interactive
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gnupg2 \
    gcc \
    g++ \
    openssh-client \
    software-properties-common \
    wget \
    git \
    python3 \
    python3-pip \
    python3-tk \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    python3-distutils
RUN apt-get install -qq libglu1-mesa-dev libx11-xcb-dev '^libxcb*'
ENV QT_QPA_PLATFORM=xcb

# install miniforge conda for mamba
WORKDIR /setup
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b
RUN rm Miniforge3-$(uname)-$(uname -m).sh
ENV PATH="/root/miniforge3/bin:${PATH}"
RUN conda init bash


# download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
#copy the conda environment file to the container and install the environment
#if the host has nvidia-docker installed, copy the gpu environment file
#COPY environment-gpu.yml /setup
#RUN --mount=type=ssh conda env create -f environment-gpu.yml
#RUN rm environment-gpu.yml

#otherwise, copy the cpu environment file
COPY environment-cpu.yml /setup
RUN --mount=type=ssh conda env create -f environment-cpu.yml
RUN rm environment-cpu.yml

#enable colors in bash
ENV TERM=xterm-256color
RUN echo "PS1='\e[92m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc
#enable directory colors
ENV LS_OPTIONS='--color=auto'
RUN echo "eval \"\$(dircolors)\"" >> /root/.bashrc
RUN echo "alias ls='ls \$LS_OPTIONS'" >> /root/.bashrc
RUN echo "alias ll='ls \$LS_OPTIONS -l'" >> /root/.bashrc
RUN echo "alias l='ls \$LS_OPTIONS -lA'" >> /root/.bashrc

RUN echo "conda activate cpx" >> ~/.bashrc
# Set the default command to run a bash shell when starting the container
WORKDIR /workspace
CMD ["bash"]
