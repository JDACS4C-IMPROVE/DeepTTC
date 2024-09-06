#!/bin/bash --login

set -e

# Run this on Lambda
# source /etc/profile.d/lmod.sh
# which nvcc
# module avail
# module load cuda/11.8
# which nvcc

# CONDA_ENV_NAME=deepttc_py37 # Conda env name
# conda create -n $CONDA_ENV_NAME python=3.7 pip --yes # Create conda env
# conda activate $CONDA_ENV_NAME # Activate conda env

# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --yes
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch

conda install -c conda-forge matplotlib --yes # installed with candle
conda install -c conda-forge h5py=3.1.0 --yes

conda install -c bioconda pubchempy=1.0.4 --yes
conda install -c rdkit rdkit=2020.09.1.0 --yes # also installs pandas
conda install -c anaconda networkx=2.6.3 --yes
# conda install -c conda-forge pyarrow=10.0 --yes
# conda install -c conda-forge pyarrow=8.0.0 --yes
conda install conda-forge::pyarrow  # potential issues!?
conda install conda install conda-forge::prettytable

conda install -c pyston psutil --yes

pip install subword-nmt
# Install CANDLE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
