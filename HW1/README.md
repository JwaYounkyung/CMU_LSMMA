
# LSMMA HW1
A repository for the assignment HW1 in LSMMA 2022 Fall.


## Library Install 
Download OpenSMILE 3.0 from https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz

conda env create -f environment.yaml
tar -zxvf opensmile-3.0-linux-x64.tar.gz
apt install ffmpeg
pip install -e 'git+https://github.com/kkoutini/ba3l@v0.0.2#egg=ba3l'
pip install -e 'git+https://github.com/kkoutini/sacred@v0.0.1#egg=sacred' 
pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.17#egg=hear21passt' 

## Run
1. `extract_soundnet_feats_passt.py`: extract snf using PaSST
2. `train_mlp_snf.py` : train
3. `test_mlp_snf.py` : test
