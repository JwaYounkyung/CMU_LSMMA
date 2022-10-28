#!/bin/python

import argparse
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

import sys
from stages import LoadFeature
from sklearn import preprocessing

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("--audio_feat_dir", default="data/passt/") #
parser.add_argument("--video_feat_dir", default="data/cnn3d_1/") #
parser.add_argument("--feat_dim", type=int, default=527)
parser.add_argument("--list_videos", default="data/labels/train_val.csv")

parser.add_argument("--audio_feat_appendix", default=".mp3.csv") 
parser.add_argument("--video_feat_appendix", default=".pkl") 

parser.add_argument("--output_file", default="weights/earlyfusion.mlp.model") #


if __name__ == '__main__':

  args = parser.parse_args()
  fread = open(args.list_videos, "r")
  
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

# %% Audio Feature

  feat_list = []
  label_list = [] # labels are [0-9]

  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.audio_feat_dir, video_id + args.audio_feat_appendix)

    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
      label_list.append(int(df_videos_label[video_id]))
    else:
      feat_list.append(np.zeros((args.feat_dim,), dtype="float"))
      label_list.append(int(df_videos_label[video_id]))

  #feat_list = preprocessing.normalize(feat_list, norm='l2')

  with open('audio_features.npy', 'wb') as f:
    np.save(f, feat_list)

  with open('labels.npy', 'wb') as f:
    np.save(f, label_list)

# %% Video Feature
  fread = open(args.list_videos, "r")
  feat_list = []

  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    video_feat_filepath = os.path.join(args.video_feat_dir, video_id + args.video_feat_appendix)
    if os.path.exists(video_feat_filepath):
      frame_features = np.stack(LoadFeature.load_features(video_feat_filepath))
      feature = np.max(frame_features, axis=0)
      feat_list.append(feature)

  #feat_list = preprocessing.normalize(feat_list, norm='l2')

  with open('video_features.npy', 'wb') as f:
    np.save(f, feat_list)

# %% Ensemble Features

  fusion_list = []
  with open('audio_features.npy', 'rb') as f:
    audio_feat_list = np.load(f)
  with open('video_features.npy', 'rb') as f:
    video_feat_list = np.load(f)
  with open('labels.npy', 'rb') as f:
    label_list = np.load(f)

  for i in range(len(audio_feat_list)):
    fusion = np.concatenate([audio_feat_list[i], video_feat_list[i]])
    fusion_list.append(fusion)

# %% Run MLP
  #1. Train a MLP classifier using feat_list and label_list
  # below are the initial settings you could use
  # hidden_layer_sizes=(512),activation="relu",solcer="adam",alpha=1e-3
  # your model should be named as "clf" to match the variable in pickle.dump()
  clf = MLPClassifier(hidden_layer_sizes=(1024),activation="relu",solver="adam",alpha=1e-4, verbose=True)
  clf.fit(fusion_list, label_list)

  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
