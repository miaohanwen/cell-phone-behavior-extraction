import argparse
import os
import numpy as np
import cv2
import collections
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
import videotransforms
from dataset import IVBSSDataset
from model import TAL_Net
import matplotlib.pyplot as plt
import random
import pickle
from collections import defaultdict


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cabin_video_dir', type=str, help='directory of cabin video')
    parser.add_argument('--face_video_dir', type=str, help='directory of face video')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--clip_length', default=16, type=int, help='Number of frames in each clip')
    parser.add_argument('--clip_stride', default=4, type=int, help='Number of frames between the starts of two clips')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of clips to process together each time')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    parser.add_argument('--threshold', type=float, help='threshold for start scores and end scores')
    #     parser.add_argument('--video_clip_file', help='extract the test video list')
    parser.add_argument('--event_file', help='path to the file containing event info')
    args = parser.parse_args()
    return args


def clip_generation(cabin_video_path, face_video_path, clip_length, clip_stride):
    cabin_frames = os.listdir(cabin_video_path)
    cabin_frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    face_frames = os.listdir(face_video_path)
    face_frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    cabin_frame_length = len(cabin_frames)
    face_frame_length = len(face_frames)
    cabin_indices = np.arange(start=0, stop=cabin_frame_length - clip_length + 1, step=clip_stride)
    face_indices = np.arange(start=0, stop=face_frame_length - clip_length * 5 + 1, step=clip_stride * 5)
    indices_in_cabin_clips = [list(range(_idx, _idx + clip_length)) for _idx in cabin_indices]
    indices_in_face_clips = [list(range(_idx, _idx + clip_length * 5)) for _idx in face_indices]
    if len(indices_in_face_clips) < len(indices_in_cabin_clips):
        indices_in_face_clips.append(range(face_frame_length - clip_length * 5, face_frame_length))
    elif len(indices_in_face_clips) > len(indices_in_cabin_clips):
        indices_in_cabin_clips.append(range(cabin_frame_length - clip_length, fcabin_frame_length))

    cabin_clips = []
    for indices_in_clip in indices_in_cabin_clips:
        clip = [cabin_frames[i] for i in indices_in_clip]
        cabin_clips.append(clip)
    face_clips = []
    for indices_in_clip in indices_in_face_clips:
        clip = [face_frames[i] for i in indices_in_clip]
        face_clips.append(clip)
    return cabin_clips, face_clips, indices_in_cabin_clips


def load_rgb_frames(video_path, clip):
    imgs = []
    for frame in clip:
        frame_path = os.path.join(video_path, frame)
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w, h, c = img.shape
        sc = float(256) / min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        imgs.append(img)
    imgs = np.asarray(imgs)
    return imgs


def predict_events(cabin_video_path, face_video_path, args):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    checkpoint = args.checkpoint
    clip_length = args.clip_length
    clip_stride = args.clip_stride
    batch_size = args.batch_size
    num_classes = args.num_classes
    threshold = args.threshold

    cabin_clips, face_clips, indices_in_cabin_clips = clip_generation(cabin_video_path, face_video_path, clip_length,
                                                                      clip_stride)
    model = TAL_Net(num_classes)
    ckp = torch.load(checkpoint)
    model.load_state_dict(ckp['model'])
    model.to(device)
    model.eval()

    clip_transforms = transforms.Compose([videotransforms.CenterCrop(224),
                                          videotransforms.ToTensor(),
                                          videotransforms.ClipNormalize()
                                          ])
    all_clips = []
    all_predict_classes = []
    all_start_scores = []
    all_end_scores = []

    n = len(cabin_clips) // batch_size
    for i in range(n):
        cabin_video_frames_batch = []
        face_video_frames_batch = []
        for j in range(i * batch_size, (i + 1) * batch_size):
            cabin_clip = cabin_clips[j]
            cabin_video_frames = load_rgb_frames(cabin_video_path, cabin_clip)
            cabin_video_frames = clip_transforms(cabin_video_frames)
            cabin_video_frames_batch.append(cabin_video_frames)
            face_clip = face_clips[j]
            face_video_frames = load_rgb_frames(face_video_path, face_clip)
            face_video_frames = clip_transforms(face_video_frames)
            face_video_frames_batch.append(face_video_frames)
        cabin_video_frames_batch = torch.stack(cabin_video_frames_batch)
        face_video_frames_batch = torch.stack(face_video_frames_batch)

        cabin_video_frames_batch = cabin_video_frames_batch.to(device)
        face_video_frames_batch = face_video_frames_batch.to(device)

        with torch.no_grad():
            class_scores, start_scores, end_scores = model(cabin_video_frames_batch, face_video_frames_batch)

        pred_classes = torch.argmax(class_scores, dim=1)
        pred_classes = pred_classes.cpu().numpy()
        start_scores = start_scores.cpu().numpy()
        end_scores = end_scores.cpu().numpy()

        all_predict_classes.append(pred_classes)
        all_start_scores.append(start_scores)
        all_end_scores.append(end_scores)

    if len(cabin_clips) % batch_size != 0:
        cabin_video_frames_batch = []
        face_video_frames_batch = []
        for k in range(n * batch_size, len(cabin_clips)):
            cabin_clip = cabin_clips[k]
            cabin_video_frames = load_rgb_frames(cabin_video_path, cabin_clip)
            cabin_video_frames = clip_transforms(cabin_video_frames)
            cabin_video_frames_batch.append(cabin_video_frames)
            face_clip = face_clips[k]
            face_video_frames = load_rgb_frames(face_video_path, face_clip)
            face_video_frames = clip_transforms(face_video_frames)
            face_video_frames_batch.append(face_video_frames)

        cabin_video_frames_batch = torch.stack(cabin_video_frames_batch)
        face_video_frames_batch = torch.stack(face_video_frames_batch)

        cabin_video_frames_batch = cabin_video_frames_batch.to(device)
        face_video_frames_batch = face_video_frames_batch.to(device)

        with torch.no_grad():
            class_scores, start_scores, end_scores = model(cabin_video_frames_batch, face_video_frames_batch)
        pred_classes = torch.argmax(class_scores, dim=1)
        pred_classes = pred_classes.cpu().numpy()
        start_scores = start_scores.cpu().numpy()
        end_scores = end_scores.cpu().numpy()

        all_predict_classes.append(pred_classes)
        all_start_scores.append(start_scores)
        all_end_scores.append(end_scores)

    all_predict_classes = np.concatenate(all_predict_classes)

    print(all_predict_classes)
    # rough chunk aggregation
    
    rough_clip_groups = defaultdict(list)
    for i in range(len(all_predict_classes)):
        if all_predict_classes[i] != 0:
            rough_clip_groups[all_predict_classes[i]].append(i)
    print(rough_clip_groups)
    all_refined_clip_groups = dict()
    for key in rough_clip_groups.keys():
        clip_group = rough_clip_groups[key]
        refined_groups = []
        
        previous = 0
        i = 0
        while i < len(clip_group) - 1:
            if clip_group[i] + 2 < clip_group[i+1]:
                refined_groups.append(clip_group[previous:(i+1)])
                previous = i+1
            i += 1
        
        refined_groups.append(clip_group[previous:])
        all_refined_clip_groups[key] = refined_groups
    print(all_refined_clip_groups)
#     all_classes = all_clip_frame_groups.keys()
    keys = list(all_refined_clip_groups)
    if len(keys) == 2:
        k1 = keys[0]
        k2 = keys[1]
        groups1 = all_refined_clip_groups[k1]
        groups2 = all_refined_clip_groups[k2]

        i = 0
        j = 0
        while i < len(groups1):
            while j < len(groups2):
                min_index1 = min(groups1[i])
                max_index1 = max(groups1[i])
                min_index2 = min(groups2[j])
                max_index2 = max(groups2[j])
                set1 = set(range(min_index1, max_index1+1))
                set2 = set(range(min_index2, max_index2+1))
                if set1.issubset(set2) == True:
                    groups1.remove(groups1[i])
                    if i >= len(groups1):
                        break
                elif set2.issubset(set1) == True:
                    groups2.remove(groups2[j])
                else:
                    if max_index1 > max_index2:
                        j += 1
                    else:
                        break
            i += 1
        final_all_clip_groups = {
            k1:groups1,
            k2:groups2
        }
    else:
        final_all_clip_groups = all_refined_clip_groups
    print(final_all_clip_groups)
    all_clip_frame_groups = {} 
    for key in final_all_clip_groups.keys():
        final_groups = final_all_clip_groups[key]
        clip_frame_groups = []
        for group in final_groups:
            clip_frame_group = set()
            for index in group:
                clip_frame_group = clip_frame_group.union(set(indices_in_cabin_clips[index]))
                start_frame = min(clip_frame_group) + 1
                end_frame = max(clip_frame_group) + 1            
            clip_frame_groups.append([start_frame, end_frame])
        all_clip_frame_groups[key] = clip_frame_groups
    return all_clip_frame_groups


def main():
    args = get_parse()
    cabin_video_dir = args.cabin_video_dir
    face_video_dir = args.face_video_dir
    event_file = args.event_file

    with open(event_file, 'rb') as f:
        event_dict = pickle.load(f)

    selected_cabin_videos = []
    for cabin_video in event_dict.keys():
        events_in_video = event_dict[cabin_video]
        num_events = 0
        for key in events_in_video.keys():
            num_events += len(events_in_video[key])
        if num_events > 2:
            selected_cabin_videos.append(cabin_video)

    print(len(selected_cabin_videos))

    all_predict_accuracy = defaultdict(list)
    for cabin_video in selected_cabin_videos:
        print(cabin_video)
        items = cabin_video.split('_', 1)
        face_video = 'Face_' + items[1]
        cabin_video_path = os.path.join(cabin_video_dir, cabin_video)
        face_video_path = os.path.join(face_video_dir, face_video)
        all_clip_frame_groups = predict_events(cabin_video_path, face_video_path, args)
        video_event_dict = event_dict[cabin_video]

        print(video_event_dict)
        print(all_clip_frame_groups)
        
        for key in all_clip_frame_groups:
            predicted_events = all_clip_frame_groups[key]
            GT_events = video_event_dict[key] 
            i = 0
            j = 0
            total_overlapped_length = 0
            while i < len(predicted_events):
                while j < len(GT_events):
                    predicted_event = predicted_events[i]
                    GT_event = GT_events[j]
                    predict_start = predicted_event[0]
                    predict_end = predicted_event[1]
                    GT_start = GT_event[0]
                    GT_end = GT_event[1]
                    
                    set1 = set(range(predict_start, predict_end + 1))
                    set2 = set(range(GT_start, GT_end + 1))
                    intersection = set1.intersection(set2)
                    overlapped_length = len(intersection)
                    total_overlapped_length += overlapped_length
                    if GT_end <= predict_end:     
                        j += 1
                    if GT_end > predict_end:
                        break           
                i += 1
                
            total_predict_length = 0
            for k in range(len(predicted_events)):
                predict_event = predicted_events[k]
                predict_start = predict_event[0]
                predict_end = predict_event[1]
                total_predict_length += predict_end - predict_start +1
                    
            predict_accuracy = float(total_overlapped_length) / (total_predict_length+1e-10)
            print(key)
            print(predict_accuracy)
            all_predict_accuracy[key].append(predict_accuracy)
    
    average_predict_accuracy = dict()       
    for key in all_predict_accuracy.keys():
        average_predict_accuracy[key] = sum(all_predict_accuracy[key])/len(all_predict_accuracy[key])
    
            
            
#         for key in video_event_dict:
#             GT_events = video_event_dict[key]
#             predicted_events = all_clip_frame_groups[key]
#             i = 0
#             j = 0
#             total_covered_length = 0
#             while i < len(predicted_events):
#                 while j < len(GT_events):
#                     predicted_event = predicted_events[i]
#                     GT_event = GT_events[j]
#                     predict_start = predicted_event[0]
#                     predict_end = predicted_event[1]
#                     GT_start = GT_event[0]
#                     GT_end = GT_event[1]
                    
#                     set1 = set(range(predict_start, predict_end + 1))
#                     set2 = set(range(GT_start, GT_end + 1))
#                     intersection = set1.intersection(set2)
#                     covered_length = len(intersection)
#                     total_covered_length += covered_length
#                     if GT_end <= predict_end:     
#                         j += 1
#                     if GT_end > predict_end:
#                         break           
#                 i += 1
                
#             total_predict_length = 0
#             total_GT_length = 0
#             for i in range(len(predicted_events)):
#                 predict_event = predicted_events[i]
#                 predict_start = predict_event[0]
#                 predict_end = predict_event[1]
#                 total_predict_length += predict_end - predict_start +1
#             for j in range(len(GT_events)):
#                 GT_event = GT_events[j]
#                 GT_start = GT_event[0]
#                 GT_end = GT_event[1]
#                 total_GT_length += GT_end - GT_start +1
                    
#             GT_covered_percent = float(total_covered_length) / (total_GT_length+1e-10)
#             predict_useful_percent = float(total_covered_length) / (total_predict_length+1e-10)
#             print(key)
#             print(GT_covered_percent, predict_useful_percent)
#             all_covered_percent[key].append(GT_covered_percent)
#             all_useful_percent[key].append(predict_useful_percent)
    
#     average_covered_percent = dict()
#     for key in all_covered_percent.keys():
#         average_covered_percent[key] = sum(all_covered_percent[key])/len(all_covered_percent[key])
#     average_useful_percent = dict()
#     for key in all_useful_percent.keys():
#         average_useful_percent[key] = sum(all_useful_percent[key])/len(all_useful_percent[key])
    
    print(average_predict_accuracy)

if __name__ == '__main__':
    main()