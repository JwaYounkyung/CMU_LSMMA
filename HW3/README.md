# CMU 11-775 Fall 2022 Homework 3

## Install Dependencies

A set of dependencies is listed in [environment.yml](environment.yml). You can use `conda` to create and activate the environment easily.

```bash
# Start from within this repo
conda env create -f environment.yml -p ./env
conda activate ./env
```

## Dataset

I used extracted features from HW1 and HW2. 1 type of audio data and 2 types of video data. You should save these data in './data'.

```bash
# Start from within this repo
cd ./data
```

Eventually, the directory structure should look like this:

* this repo
  * code
  * data
    * labels 
    * passt (unzipped from passt.zip)
    * cnn (unzipped from cnn.zip)
    * cnn3d (unzipped from cnn3d.zip)
  * env
  * featuress
  * result
  * weights
  * environment.yml
  * README.md


## Preprocessing

To get numpy array of features, use

```bash
python code/preprocessing --audio_feat_dir data/passt/ --video_feat_dir data/cnn/ --video_feat_name cnn
```
The numpy arrays are stored under `features/`.

## Early Fusion

By default, preprocessed features are stored under `features/`.

To train and test earlyfusion model, use

```bash
python code/earlyfusion_train.py --video_feat_name cnn3d
python code/earlyfusion_test.py --video_feat_name cnn3d
```
By default, model weights are stored under `weights/`.


## Late Fusion

By default, preprocessed features are stored under `features/`.

To train and test latefusion model, use

```bash
python code/latefusion_train.py --video_feat_name cnn3d
python code/latefusion_test.py --video_feat_name cnn3d
```
By default, model weights are stored under `weights/`.
By default, test results are stored under `results/`.

## Double Fusion

By default, preprocessed features are stored under `features/`.

To train and test doublefusion model, use

```bash
python code/earlyfusion_train.py --video_feat_name cnn3d
python code/earlyfusion_train.py --video_feat_name cnn

python code/doublefusion_test.py --video_feat_name1 cnn3d --video_feat_name2 cnn
```
By default, model weights are stored under `weights/`.
By default, test results are stored under `results/`.

