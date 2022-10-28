#!/bin/python

import argparse
from operator import not_
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

from stages import LoadFeature

# Apply the MLP model to the testing videos;
# Output prediction class for each video

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--early1_model_file", default="weights/earlyfusion.mlp.model") #
  parser.add_argument("--early2_model_file", default="weights/earlyfusion2.mlp.model") #
  
  parser.add_argument("--audio_feat_dir", default="data/passt/") 
  parser.add_argument("--list_videos", default="data/labels/test_for_students.csv")

  parser.add_argument("--output_file", default="doublefusion.mlp.csv")

  parser.add_argument("--audio_feat_appendix", default=".mp3.csv") 
  parser.add_argument("--video_feat_appendix", default=".pkl") 

  return parser.parse_args()


if __name__ == '__main__':

  args = parse_args()

  fusion1_list, fusion2_list = [], []
  video_ids = []
  not_found_count = 0

  for line in open(args.list_videos).readlines()[1:]:
    video_id= line.strip().split(",")[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.audio_feat_dir, video_id + args.audio_feat_appendix)

  with open('audio_features.test.npy', 'rb') as f:
    audio_feat_list = np.load(f)
  with open('video_features.test.npy', 'rb') as f:
    video_feat1_list = np.load(f)
  with open('video_features2.test.npy', 'rb') as f:
    video_feat2_list = np.load(f)
  
  for i in range(len(audio_feat_list)):
    fusion = np.concatenate([audio_feat_list[i], video_feat1_list[i]])
    fusion1_list.append(fusion)
  for i in range(len(audio_feat_list)):
    fusion = np.concatenate([audio_feat_list[i], video_feat2_list[i]])
    fusion2_list.append(fusion)

  # Load model and get predictions
  # the shape of pred_classes should be (num_samples)
  early1_clf = pickle.load(open(args.early1_model_file, 'rb'))
  early1_pred = early1_clf.predict_proba(fusion1_list)

  early2_clf = pickle.load(open(args.early2_model_file, 'rb'))
  early2_pred = early1_clf.predict_proba(fusion2_list)

  fusion_pred = np.mean(np.array([early1_pred, early2_pred]), axis=0)
  pred_classes = np.argmax(fusion_pred, axis=-1)

  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
