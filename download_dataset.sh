#!/bin/bash

#if StaticArm04_02.db does not exist inside data directory, download it
if [ ! -f data/StaticArm04_02.db ]; then
    #check if python interpreter has gdown installed
    if ! python3 -c "import gdown" &> /dev/null; then
        echo "gdown not installed, installing..."
        pip3 install gdown
    fi
    python3 -m gdown 1v5AMVgD4ygHfaBYB6yt49vSTpC9s1mF2 -O data/StaticArm04_02.db
else
    echo "StaticArm04_02.db already exists in data directory"
fi
#check if ROBDATA_PATH env is set, else set it to the data directory
if [ -z "$ROBDATA_PATH" ]; then
    echo "export ROBDATA_PATH=$(pwd)/data" >> ~/.bashrc
    source ~/.bashrc
    echo "ROBDATA_PATH environment variable set to $(pwd)/data and added to ~/.bashrc"
else
    echo "ROBDATA_PATH environment variable already set"
fi
