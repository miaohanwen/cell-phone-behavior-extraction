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
import time


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
#     parser.add_argument('--event_file', help='path to the file containing event info')
    args = parser.parse_args()
    return args


def clip_generation(cabin_video_path, face_video_path, clip_length, clip_stride):
    cabin_frames = os.listdir(cabin_video_path)
    cabin_frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    face_frames = os.listdir(face_video_path)
    face_frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    cabin_frame_length = len(cabin_frames)
    face_frame_length = len(face_frames)
    
    if cabin_frame_length < clip_length:
        indices_in_cabin_clip = list(range(cabin_frame_length))
        indices_in_cabin_clip += [cabin_frame_length-1]*(clip_length-cabin_frame_length)
        indices_in_cabin_clips = [indices_in_cabin_clip]
        indices_in_face_clip = list(range(face_frame_length))
        indices_in_face_clip += [face_frame_length-1]*(clip_length*5-face_frame_length)
        indices_in_face_clips = [indices_in_face_clip]
    else:
        cabin_indices = np.arange(start=0, stop=cabin_frame_length - clip_length + 1, step=clip_stride)
        face_indices = np.arange(start=0, stop=face_frame_length - clip_length * 5 + 1, step=clip_stride * 5)
        indices_in_cabin_clips = [list(range(_idx, _idx + clip_length)) for _idx in cabin_indices]
        indices_in_face_clips = [list(range(_idx, _idx + clip_length * 5)) for _idx in face_indices]
        if len(indices_in_face_clips) < len(indices_in_cabin_clips): 
            indices_in_face_clips.append(range(face_frame_length - clip_length * 5, face_frame_length))
        elif len(indices_in_face_clips) > len(indices_in_cabin_clips):
            del indices_in_face_clips[-1]
#         indices_in_cabin_clips.append(range(cabin_frame_length - clip_length, fcabin_frame_length))

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
    all_start_scores = np.concatenate(all_start_scores)
    all_end_scores = np.concatenate(all_end_scores)
#     print(all_predict_classes)
    
    # refined chunk aggregation
    cabin_frames = os.listdir(cabin_video_path)
    cabin_frame_length  = len(cabin_frames)
    cabin_indices = np.arange(start=0, stop=cabin_frame_length - clip_stride + 1, step=clip_stride)
    
    if len(all_predict_classes) == 1:
        label = all_predict_classes[0]
        return {label: [0, cabin_frame_length-1]}
   
    indices_in_shorter_clips = [list(range(idx, idx + clip_stride)) for idx in cabin_indices]
#     remainder = cabin_frame_length % clip_stride
#     if remainder != 0:
#         indices_in_shorter_clips.append(list(range(cabin_frame_length-remainder, cabin_frame_length)))
#     print(len(indices_in_shorter_clips))
#     print(len(indices_in_cabin_clips))
    shorter_clip_predict_classes = []
     
        
    for i in range(len(indices_in_shorter_clips)):
        start_idx = max(0, i-3)
        end_idx = min(i+1, len(all_predict_classes))
        l = [all_predict_classes[j] for j in range(start_idx, end_idx)]
        shorter_clip_predict_classes.append(max(set(l), key = l.count))
    
#     for i in range(len(indices_in_shorter_clips)):
#         if i == 0:
#             shorter_clip_predict_classes.append(all_predict_classes[0])
#         elif i == 1:
#             l = [all_predict_classes[0], all_predict_classes[1]]
#             shorter_clip_predict_classes.append(max(set(l), key = l.count))
#         elif i == 2:
#             l = [all_predict_classes[0], all_predict_classes[1], all_predict_classes[2]]
#             shorter_clip_predict_classes.append(max(set(l), key = l.count))
#         elif i < len(indices_in_cabin_clips):
#             l = [all_predict_classes[j] for j in range(i-3, i+1)]
#             shorter_clip_predict_classes.append(max(set(l), key = l.count))
#         elif i == len(indices_in_cabin_clips):
#             index = len(indices_in_cabin_clips) - 1
#             l = [all_predict_classes[index-2], all_predict_classes[index-1], all_predict_classes[index]]
#             shorter_clip_predict_classes.append(max(set(l), key = l.count))
#         elif i == len(indices_in_cabin_clips) + 1:
#             index = len(indices_in_cabin_clips) - 1
#             l = [all_predict_classes[index-1], all_predict_classes[index]]
#             shorter_clip_predict_classes.append(max(set(l), key = l.count))
#         elif i == len(indices_in_cabin_clips) + 2:
#             index = len(indices_in_cabin_clips) - 1
#             shorter_clip_predict_classes.append(all_predict_classes[index])
#     print(shorter_clip_predict_classes)
    
    # extract start and end peaks
    start_peak_indices = []
    end_peak_indices = []
    if all_start_scores[0] > all_start_scores[1]:
        start_peak_indices.append(0)
    for i in range(1, len(all_start_scores) - 1):
        if all_start_scores[i] > all_start_scores[i - 1]:
            if all_start_scores[i] > all_start_scores[i + 1]:
                start_peak_indices.append(i)
        if all_end_scores[i] > all_end_scores[i - 1]:
            if all_end_scores[i] > all_end_scores[i + 1]:
                end_peak_indices.append(i)
    if all_end_scores[-1] > all_end_scores[-2]:
        end_peak_indices.append(len(cabin_clips) - 1)

    j = 0
    copy_start_peak_indices = start_peak_indices.copy()
    while j < len(start_peak_indices) - 1:
        index1 = copy_start_peak_indices[j]
        index2 = copy_start_peak_indices[j + 1]
        if index1 + 4 < index2:
            j += 1
        else:
            if all_start_scores[start_peak_indices[j]] > all_start_scores[start_peak_indices[j + 1]]:
                copy_start_peak_indices[j] = index2
                copy_start_peak_indices.pop(j + 1)
                start_peak_indices.pop(j + 1)

            else:
                copy_start_peak_indices.pop(j)
                start_peak_indices.pop(j)

    k = 0
    copy_end_peak_indices = end_peak_indices.copy()
    while k < len(end_peak_indices) - 1:
        index1 = copy_end_peak_indices[k]
        index2 = copy_end_peak_indices[k + 1]
        if index1 + 4 < index2:
            k += 1
        else:
            if all_end_scores[end_peak_indices[k]] > all_end_scores[end_peak_indices[k + 1]]:
                copy_end_peak_indices[k] = index2
                copy_end_peak_indices.pop(k + 1)
                end_peak_indices.pop(k + 1)
            else:
                copy_end_peak_indices.pop(k)
                end_peak_indices.pop(k)

    selected_starts = []
    selected_ends = []
    for start_indice in start_peak_indices:
        if all_start_scores[start_indice] > threshold:
            selected_starts.append(start_indice)
    for end_indice in end_peak_indices:
        if all_end_scores[end_indice] > threshold:
            selected_ends.append(end_indice+3)
#     print(selected_starts)
#     print(selected_ends)
    
       
    rough_clip_groups = defaultdict(list)
    for i in range(len(shorter_clip_predict_classes)):
        if shorter_clip_predict_classes[i] != 0:
            rough_clip_groups[shorter_clip_predict_classes[i]].append(i)
#     print(rough_clip_groups)
    
    all_refined_clip_groups = dict()
    
    for key in rough_clip_groups.keys():
        clip_group = rough_clip_groups[key]
        refined_groups = []
        previous = 0
        i = 0
        while i < len(clip_group) - 1:
            if clip_group[i+1] - clip_group[i] < 4:
                if len(set(range(clip_group[i], clip_group[i+1])).intersection(set(selected_starts))) != 0 or len(set(range(clip_group[i], clip_group[i+1])).intersection(set(selected_ends))) != 0:
                    if len(clip_group[previous:(i+1)]) > 1:
                        refined_groups.append(clip_group[previous:(i+1)])
                    previous = i+1
            elif clip_group[i+1] - clip_group[i] >= 4:
                if len(clip_group[previous:(i+1)]) > 1:
                    refined_groups.append(clip_group[previous:(i+1)])
                previous = i+1
            i += 1
            
        if len(clip_group[previous:]) > 1:
            refined_groups.append(clip_group[previous:])
        j = 0
        while j < len(refined_groups)-1:
            if refined_groups[j+1][0] - refined_groups[j][-1] == 1:
                refined_groups[j] += refined_groups[j+1]
                refined_groups.remove(refined_groups[j+1])
            else:
                j += 1
        all_refined_clip_groups[key] = refined_groups
#     print(all_refined_clip_groups)

    
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
        filtered_all_clip_groups = {}
        if groups1 != []:
            filtered_all_clip_groups[k1] = groups1
        if groups2 != []:
            filtered_all_clip_groups[k2] = groups2
    else:
        filtered_all_clip_groups = all_refined_clip_groups
#     print(filtered_all_clip_groups)
    
    all_clip_frame_groups = {} 
    for key in filtered_all_clip_groups.keys():
        final_groups = filtered_all_clip_groups[key]
        clip_frame_groups = []
        for group in final_groups:
            clip_frame_group = set()
            for index in group:
                clip_frame_group = clip_frame_group.union(set(indices_in_shorter_clips[index]))
                start_frame = min(clip_frame_group) + 1
                end_frame = max(clip_frame_group) + 1            
            clip_frame_groups.append([start_frame, end_frame])
            
        
        all_clip_frame_groups[key] = clip_frame_groups
    return all_clip_frame_groups


def main():
    args = get_parse()
    cabin_video_dir = args.cabin_video_dir
    face_video_dir = args.face_video_dir
    
    cabin_video_list = os.listdir(cabin_video_dir)
    
    random.seed(0)
    selected_cabin_videos = random.sample(cabin_video_list, 50)
    
    total_duration = 0.0
    for cabin_video in selected_cabin_videos:
        frame_list = os.listdir(os.path.join(cabin_video_dir, cabin_video))
            
        duration = len(frame_list)/2
        total_duration += duration
    print(total_duration)
    
    start_time = time.time()
    
    for cabin_video in selected_cabin_videos:
        print(cabin_video)
        items = cabin_video.split('_', 1)
        face_video = 'Face_' + items[1]
        cabin_video_path = os.path.join(cabin_video_dir, cabin_video)
        face_video_path = os.path.join(face_video_dir, face_video)
        all_clip_frame_groups = predict_events(cabin_video_path, face_video_path, args)
        print(all_clip_frame_groups)
    
    end_time = time.time()
    running_time = end_time-start_time
    running_time_per_second = running_time / total_duration
    print(running_time_per_second)


if __name__ == '__main__':
    main()