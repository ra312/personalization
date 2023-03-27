#!/bin/bash

# Generate a UUID for the kernel name
uuid=$(uuidgen)

# Set the name of your new kernel
env_name="model-server-$uuid"

# Create local virtual env
python -m venv "${env_name}"

# Activate your virtual environment (if necessary)

source activate "${env_name}"

# Install ipykernel package
pip install ipykernel

# Register your kernel
python -m ipykernel install --user --name "$env_name" --display-name "Model Server"

