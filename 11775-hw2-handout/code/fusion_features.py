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
parser.add_argument("--video_feat_dir", default="data/cnn/") #
parser.add_argument("--feat_dim", type=int, default=527)
parser.add_argument("--train_list_videos", default="data/labels/train_val.csv")
parser.add_argument("--test_list_videos", default="data/labels/test_for_students.csv")

parser.add_argument("--audio_feat_appendix", default=".mp3.csv") 
parser.add_argument("--video_feat_appendix", default=".pkl") 

parser.add_argument("--output_file", default="weights/earlyfusion.mlp.model") #


if __name__ == '__main__':

  args = parser.parse_args()
  
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.train_list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

# %% Audio Train Feature

  feat_list = []
  label_list = [] # labels are [0-9]

  for line in open(args.train_list_videos).readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.audio_feat_dir, video_id + args.audio_feat_appendix)

    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
      label_list.append(int(df_videos_label[video_id]))
    else:
      feat_list.append(np.zeros((args.feat_dim,), dtype="float"))
      label_list.append(int(df_videos_label[video_id]))

  #feat_list = preprocessing.normalize(feat_list, norm='l2')

  with open('audio_features.train.npy', 'wb') as f:
    np.save(f, feat_list)

  with open('labels.npy', 'wb') as f:
    np.save(f, label_list)

# %% Video Train Feature
  feat_list = []

  for line in  open(args.train_list_videos).readlines()[1:]:
    video_id = line.strip().split(",")[0]
    video_feat_filepath = os.path.join(args.video_feat_dir, video_id + args.video_feat_appendix)
    if os.path.exists(video_feat_filepath):
      frame_features = np.stack(LoadFeature.load_features(video_feat_filepath))
      feature = np.max(frame_features, axis=0)
      feat_list.append(feature)

  #feat_list = preprocessing.normalize(feat_list, norm='l2')

  with open('video_features2.train.npy', 'wb') as f:
    np.save(f, feat_list)

# %% Test Features
  audio_feat_list, video_feat_list = [], []
  fusion_list = []
  video_ids = []
  not_found_count = 0

  for line in open(args.test_list_videos).readlines()[1:]:
    video_id= line.strip().split(",")[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.audio_feat_dir, video_id + args.audio_feat_appendix)

    if not os.path.exists(feat_filepath):
      audio_feat_list.append(np.zeros(args.audio_feat_dim))
      not_found_count += 1
    else:
      audio_feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
  
  for line in open(args.test_list_videos).readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.video_feat_dir, video_id + args.video_feat_appendix)
    # for videos with no audio, ignored in training
    if not os.path.exists(feat_filepath):
      video_feat_list.append(np.zeros(args.video_feat_dim))
      not_found_count += 1
    else:
      frame_features = np.stack(LoadFeature.load_features(feat_filepath))
      feature = np.max(frame_features, axis=0)
      video_feat_list.append(feature)
  
  if not_found_count > 0:
    print(f'Could not find the features for {not_found_count} samples.')

  with open('audio_features.test.npy', 'wb') as f:
    np.save(f, audio_feat_list)
  with open('video_features2.test.npy', 'wb') as f:
    np.save(f, video_feat_list)
  