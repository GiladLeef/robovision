#!/bin/bash

# Install pre-requisites
sudo apt install -y curl git libssl-dev libffi-dev

# Install pyenv
curl -fsSL https://pyenv.run | bash

# Add pyenv to bash profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

# Source the profile to update PATH
source ~/.bashrc

# Install Python 3.10.9
pyenv install 3.10.9
pyenv global 3.10.9

# Create virtual environment
mkdir -p /opt
pyenv virtualenv 3.10.9 /opt/venv

# Activate the virtual environment
source /opt/venv/bin/activate

# Install libraries
pip install onnxruntime numpy python-opencv

deactivate

echo "robovision dependencies installed successfully."
echo "Run using:"
echo "source /opt/venv/bin/activate"
echo "python /opt/robovision/webcamntv.py"
