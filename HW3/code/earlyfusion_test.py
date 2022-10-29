#!/bin/python

import argparse
from operator import not_
import os
import pickle

import numpy as np

# Apply the MLP model to the testing videos;
# Output prediction class for each video

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_file", default="weights/earlyfusion.mlp.model") #
  
  parser.add_argument("--audio_feat_dir", default="data/passt/") #
  parser.add_argument("--video_feat_name", default="cnn3d") #
  parser.add_argument("--list_videos", default="data/labels/test_for_students.csv")

  parser.add_argument("--audio_feat_appendix", default=".mp3.csv") 
  parser.add_argument("--video_feat_appendix", default=".pkl") 

  parser.add_argument("--output_file", default="results/doublefusion.mlp.csv")

  return parser.parse_args()


if __name__ == '__main__':

  args = parse_args()

  audio_feat_list, video_feat_list = [], []
  fusion_list = []
  video_ids = []

  for line in open(args.list_videos).readlines()[1:]:
    video_id= line.strip().split(",")[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.audio_feat_dir, video_id + args.audio_feat_appendix)

  with open('features/audio_features.test.npy', 'rb') as f:
    audio_feat_list = np.load(f)
  with open('features/video_features'+args.video_feat_name+ '.train.npy', 'rb') as f:
    video_feat_list = np.load(f)


  for i in range(len(audio_feat_list)):
    fusion = np.concatenate([audio_feat_list[i], video_feat_list[i]])
    fusion_list.append(fusion)

  # Load model and get predictions
  # the shape of pred_classes should be (num_samples)
  clf = pickle.load(open(args.model_file, 'rb'))
  pred_classes = clf.predict(fusion_list)

  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
