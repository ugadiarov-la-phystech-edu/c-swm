FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
         libgl1 libglib2.0-0 git && \
     rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN pip install -q --no-cache gym==0.19.0
RUN pip install -q --no-cache gym[atari] scikit-image matplotlib h5py

ARG UNAME
ARG GID
ARG UID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
WORKDIR /workspace

ENV PATH "/opt/conda/bin:$PATH"
ENV PYTHONPATH "$PYTHONPATH:/workspace/"
ENV OMP_NUM_THREADS=1