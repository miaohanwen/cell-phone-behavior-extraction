import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import random


def load_rgb_frames(video_dir, video_id, frame_names):
    frame_dir = os.path.join(video_dir,video_id)
    imgs = []
    for frame_name in frame_names:
        frame_path = os.path.join(frame_dir, frame_name)
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w, h, c = img.shape
        sc = float(256)/min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        imgs.append(img)
    imgs = np.asarray(imgs)
    return imgs


class IVBSSDataset(Dataset):
    def __init__(self, cabin_video_dir, face_video_dir, clip_info_path, transforms=None):
        """
        Args:
        cabin_video_dir(string): Directory with all the cabin video frames.
        clip_info_path(string): Path to the json file containing information for clips.
        transform(callable, optional): Optional transform to be applied on a sample.
        """
        random.seed(1)
        self.cabin_video_dir = cabin_video_dir
        self.face_video_dir = face_video_dir
        
        with open(clip_info_path,'r') as f:
            self.all_clip_info = json.load(f)
            
        random.shuffle(self.all_clip_info)
        self.transforms = transforms
#         self.all_clip_info = self.all_clip_info[:320]
        
    def __len__(self):
        return len(self.all_clip_info)

    def __getitem__(self, idx):
        clip_info = self.all_clip_info[idx]
        cabin_video_id = clip_info['cabin_video_id']
        face_video_id = clip_info['face_video_id']
        cabin_frames = clip_info['cabin frames']
        face_frames = clip_info['face frames']
        start = clip_info['start']
        end = clip_info['end']
        label = clip_info['label']
        cabin_imgs = load_rgb_frames(self.cabin_video_dir, cabin_video_id, cabin_frames)
        face_imgs = load_rgb_frames(self.face_video_dir, face_video_id, face_frames)
        if self.transforms is not None:
            cabin_imgs = self.transforms(cabin_imgs)
            face_imgs = self.transforms(face_imgs)
#         if label == 1:
#             label = 1
#         elif label == 3:
#             label = 2
#         elif label == 5 or label == 7:
#             label = 3
        if label == 1 or label == 3:
            label = 1
        elif label == 5 or label == 7:
            label = 2
        return cabin_imgs, face_imgs, label, start, end


def collate_fn(batch):
    cabin_imgs, face_imgs, labels, starts, ends = zip(*batch)
    cabin_imgs = torch.stack(cabin_imgs)
    face_imgs = torch.stack(face_imgs)
    labels = torch.tensor(labels, dtype=torch.long)
    starts = torch.tensor(starts, dtype=torch.float)
    ends = torch.tensor(ends, dtype=torch.float)
    return cabin_imgs, face_imgs, labels, starts, ends
    














