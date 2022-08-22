FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --allow-unauthenticated wget vim less git build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/{apt,dpkg,cache,log}/
RUN ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

ARG UNAME
ARG GID
ARG UID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
WORKDIR /home/$UNAME

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /home/$UNAME/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ADD .mujoco /home/$UNAME/.mujoco
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$UNAME/.mujoco/mujoco200/bin' >> /home/$UNAME/.bashrc
RUN echo 'source /home/$UNAME/miniconda3/bin/activate' >> /home/$UNAME/.bashrc

RUN git clone https://github.com/ugadiarov-la-phystech-edu/ROLL.git
WORKDIR /home/$UNAME/ROLL

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/home/$UNAME/.mujoco/mujoco200/bin
RUN . /home/$UNAME/miniconda3/bin/activate && conda env create -f environment.yml

WORKDIR /home/$UNAME/ROLL/multiworld
RUN . /home/$UNAME/miniconda3/bin/activate && conda activate ROLL && pip install .

WORKDIR /home/$UNAME
RUN rm -rf ROLL

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH /home/$UNAME/miniconda3/bin:$PATH
ENV PYTHONPATH "$PYTHONPATH:/workspace/ROLL"
ENV PJHOME "/workspace/ROLL"
ENV OMP_NUM_THREADS=1