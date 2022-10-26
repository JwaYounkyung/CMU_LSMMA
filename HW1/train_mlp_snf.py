#!/bin/python

import argparse
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.utils import shuffle


import sys

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("--feat_dir", default="snf2/")
parser.add_argument("--feat_dim", type=int, default=50)
parser.add_argument("--list_videos", default="labels/train_val.csv")
parser.add_argument("--output_file", default="weights/snf.mlp.model")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--val", default=False)

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
      label_list.append(int(df_videos_label[video_id]))
  
  # shuffle for a validation
  feat_list, label_list = shuffle(feat_list, label_list)

  # model setting
  clf = MLPClassifier(hidden_layer_sizes=(2048, 1024, 512, 256, 128),activation="relu",solver="adam",alpha=1e-4)

  if args.val: 
    train_len = int(len(feat_list)*0.7)
    train_feat = feat_list[:train_len]
    train_label = label_list[:train_len]

    val_feat = feat_list[train_len:]
    val_label = label_list[train_len:] 

    # hidden_layer_sizes=(512),activation="relu",solcer="adam",alpha=1e-3

    # train
    clf.fit(train_feat, train_label)

    # validation
    val_pred = clf.predict(val_feat)
    val_acc = metrics.accuracy_score(val_pred, val_label) 
    print('validation accuracy :', val_acc)
  else:
    # train
    clf.fit(feat_list, label_list)

    # save trained MLP in output_file
    pickle.dump(clf, open(args.output_file, 'wb'))
    print('MLP classifier trained successfully')
