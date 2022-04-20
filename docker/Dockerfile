FROM ubuntu:18.04


ARG PYTHON_VERSION=3.9


# Install necessary building tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         sudo \
         bzip2 \
         libx11-6 \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y dist-upgrade && apt-get purge -y libboost-all-dev && \
    apt-get install -f -y libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /app
WORKDIR /app


# Install conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
    /opt/conda/bin/conda clean -ya


ENV PATH /opt/conda/bin:$PATH

RUN apt-get update
RUN apt-get install -y vim lshw
RUN apt-get install -y ghostscript
RUN apt-get install -y poppler-utils

ENV LANG C.UTF-8

RUN export CC=mpicc && export HDF5_MPI="ON" && /opt/conda/bin/pip install versioned-hdf5

# Install data science and machine learning packages using conda
RUN /opt/conda/bin/conda install -y -c conda-forge joblib numba tqdm && \
    /opt/conda/bin/conda install -y -c conda-forge sphinx_rtd_theme recommonmark nbsphinx ipywidgets && \
    /opt/conda/bin/conda clean -ya


# Install data science and machine learning packages using conda
RUN /opt/conda/bin/conda install -y -c conda-forge scikit-learn scikit-optimize pandas seaborn && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -y -c conda-forge matplotlib">=3.4.3" && \
    /opt/conda/bin/conda install -c anaconda python-dateutil && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -y -c anaconda ujson && \
    /opt/conda/bin/conda clean -ya


# Install Jupyter
RUN /opt/conda/bin/conda install -y -c conda-forge jupyterlab jupyter_http_over_ws nodejs">=16.0.0"  && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -y -c anaconda nbformat && \
    /opt/conda/bin/conda install -y ipykernel && \
    /opt/conda/bin/conda clean -ya
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN python3 -m ipykernel.kernelspec
