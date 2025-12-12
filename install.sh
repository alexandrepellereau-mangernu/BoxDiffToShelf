sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv -y

# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc

python3 -m venv .venv
source .venv/bin/activate

# Installation
pip install -r requirements.txt
