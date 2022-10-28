#!/bin/python

import argparse
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

import sys
from stages import LoadFeature

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("--audio_feat_dir", default="data/passt/") #
parser.add_argument("--video_feat_dir", default="data/cnn3d_1/") #
parser.add_argument("--feat_dim", type=int, default=527)
parser.add_argument("--list_videos", default="data/labels/train_val.csv")

parser.add_argument("--audio_feat_appendix", default=".mp3.csv") 
parser.add_argument("--video_feat_appendix", default=".pkl") 

parser.add_argument("--audio_output_file", default="weights/latefusion.audio.mlp.model") #
parser.add_argument("--video_output_file", default="weights/latefusion.video.mlp.model") #

if __name__ == '__main__':

  args = parser.parse_args()
  fread = open(args.list_videos, "r")
  
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

# %% Ensemble Features

  with open('audio_features.npy', 'rb') as f:
    audio_feat_list = np.load(f)
  with open('video_features.npy', 'rb') as f:
    video_feat_list = np.load(f)
  with open('labels.npy', 'rb') as f:
    label_list = np.load(f)

# %% Run MLP
  #1. Train a MLP classifier using feat_list and label_list
  # below are the initial settings you could use
  # hidden_layer_sizes=(512),activation="relu",solcer="adam",alpha=1e-3
  # your model should be named as "clf" to match the variable in pickle.dump()
  audio_clf = MLPClassifier(hidden_layer_sizes=(1024),activation="relu",solver="adam",alpha=1e-4, verbose=True)
  audio_clf.fit(audio_feat_list, label_list)

  video_clf = MLPClassifier(hidden_layer_sizes=(1024),activation="relu",solver="adam",alpha=1e-4, verbose=True)
  video_clf.fit(video_feat_list, label_list)

  # save trained MLP in output_file
  pickle.dump(audio_clf, open(args.audio_output_file, 'wb'))
  pickle.dump(video_clf, open(args.video_output_file, 'wb'))
  print('MLP classifier trained successfully')
