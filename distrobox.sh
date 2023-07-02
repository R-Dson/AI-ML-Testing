#!/bin/bash
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/"
##name = 
#distrobox-enter fedora-37 -- setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#distrobox-enter fedora-37 -- conda activate tf-gpu && python Model.py
#distrobox-enter fedora-37 -- bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/"
