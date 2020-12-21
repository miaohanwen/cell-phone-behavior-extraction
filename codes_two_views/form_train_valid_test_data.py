import pandas as pd
import os
import numpy as np
import argparse
import glob
import re
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_clip_json_file')
    args = parser.parse_args()
    return args


def form_data(video_clips):
    
    for clip in video_clips:
        if clip['rst'] == -1 and clip['rend'] == 1:
            


def main():
    args = parse_args()
    video_clip_json_file = args.video_clip_json_file
    with open(video_clip_json_file, 'r') as f:
        video_clips = json.load(f)
    train_video_clips = clips['train_video_clips']
    val_video_clips = clips['val_video_clips']
    test_video_clips = clips['test_video_clips']
    


if __name__ == "__main__":
    main()

