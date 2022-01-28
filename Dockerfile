FROM pytorch/pytorch:latest

ARG TZ=Asia/Tokyo

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt update && apt install -y \
    git \
    wget \
    curl \
    unzip \
    vim \
    x11-apps \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
    build-essential cmake libclang-dev \
    && apt autoremove -y \
    && apt clean -y

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t robbyrussell

RUN mkdir /root/.vim
COPY .vimrc /root/.vimrc

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

