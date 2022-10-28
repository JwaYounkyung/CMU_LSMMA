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
  parser.add_argument("--audio_model_file", default="weights/latefusion.audio.mlp.model") #
  parser.add_argument("--video_model_file", default="weights/latefusion.video.mlp.model") #
  
  parser.add_argument("--audio_feat_dir", default="data/passt/") #
  parser.add_argument("--video_feat_dir", default="data/cnn3d_1/") #
  parser.add_argument("--audio_feat_dim", type=int, default=527)
  parser.add_argument("--video_feat_dim", type=int, default=512)
  parser.add_argument("--list_videos", default="data/labels/test_for_students.csv")

  parser.add_argument("--output_file", default="latefusion.mlp.csv")

  parser.add_argument("--audio_feat_appendix", default=".mp3.csv") 
  parser.add_argument("--video_feat_appendix", default=".pkl") 

  return parser.parse_args()


if __name__ == '__main__':

  args = parse_args()

  audio_feat_list, video_feat_list = [], []
  fusion_list = []
  video_ids = []
  not_found_count = 0

  for line in open(args.list_videos).readlines()[1:]:
    video_id= line.strip().split(",")[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.audio_feat_dir, video_id + args.audio_feat_appendix)

    if not os.path.exists(feat_filepath):
      audio_feat_list.append(np.zeros(args.audio_feat_dim))
      not_found_count += 1
    else:
      audio_feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

  for line in open(args.list_videos).readlines()[1:]:
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

  # Load model and get predictions
  # the shape of pred_classes should be (num_samples)
  audio_clf = pickle.load(open(args.audio_model_file, 'rb'))
  audio_pred = audio_clf.predict_proba(audio_feat_list)

  video_clf = pickle.load(open(args.video_model_file, 'rb'))
  video_pred = video_clf.predict_proba(video_feat_list)

  fusion_pred = np.mean(np.array([audio_pred, video_pred]), axis=0)
  pred_classes = np.argmax(fusion_pred, axis=-1)

  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
