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
        indices_in_face_clips.append(list(range(face_frame_length - clip_length * 5, face_frame_length)))
    elif len(indices_in_face_clips) > len(indices_in_cabin_clips):
        del indices_in_face_clips[-1]
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


def predict_video(cabin_video_path, face_video_path, args):
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

    cabin_clips, face_clips, indices_in_cabin_clips = clip_generation(cabin_video_path, face_video_path, clip_length, clip_stride)
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
#     print(all_start_scores)
#     print(all_end_scores)
#     start_peak_indices = []
#     end_peak_indices = []
    
#     if all_start_scores[0] > all_start_scores[1]:
#         start_peak_indices.append(0)
#     for i in range(1, len(cabin_clips) - 1):
#         if all_start_scores[i] > all_start_scores[i - 1]:
#             if all_start_scores[i] > all_start_scores[i + 1]:
#                 start_peak_indices.append(i)
#         if all_end_scores[i] > all_end_scores[i - 1]:
#             if all_end_scores[i] > all_end_scores[i + 1]:
#                 end_peak_indices.append(i)
#     if all_end_scores[-1] > all_end_scores[-2]:
#         end_peak_indices.append(len(cabin_clips) - 1)

#     j = 0
#     copy_start_peak_indices = start_peak_indices.copy()
#     while j < len(start_peak_indices) - 1:
#         index1 = copy_start_peak_indices[j]
#         index2 = copy_start_peak_indices[j + 1]
#         if index1 + 4 < index2:
#             j += 1
#         else:
#             if all_start_scores[start_peak_indices[j]] > all_start_scores[start_peak_indices[j+1]]:
#                 copy_start_peak_indices[j] = index2
#                 copy_start_peak_indices.pop(j + 1)
#                 start_peak_indices.pop(j + 1)

#             else:
#                 copy_start_peak_indices.pop(j)
#                 start_peak_indices.pop(j)

#     k = 0
#     copy_end_peak_indices = end_peak_indices.copy()
#     while k < len(end_peak_indices) - 1:
#         index1 = copy_end_peak_indices[k]
#         index2 = copy_end_peak_indices[k + 1]
#         if index1 + 4 < index2:
#             k += 1
#         else:
#             if all_end_scores[end_peak_indices[k]] > all_end_scores[end_peak_indices[k+1]]:
#                 copy_end_peak_indices[k] = index2
#                 copy_end_peak_indices.pop(k + 1)
#                 end_peak_indices.pop(k + 1)
#             else:
#                 copy_end_peak_indices.pop(k)
#                 end_peak_indices.pop(k)
                
    selected_starts = []
    selected_ends = []
    for i in range(len(all_start_scores)):
        if all_start_scores[i] > threshold:
            selected_starts.append(i)
    for j in range(len(all_end_scores)):
        if all_end_scores[j] > threshold:
            selected_ends.append(j)        
    return selected_starts, selected_ends, all_start_scores, indices_in_cabin_clips


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
    
#     all_predicted_starts_accuracy = []
#     all_predicted_ends_accuracy = []
    all_true_positive_rate = []
    all_false_positive_rate = []
    all_precision = []
    all_recall = []
    all_true_positive_rate1 = []
    all_false_positive_rate1 = []
    all_precision1 = []
    all_recall1 = []
    
    for cabin_video in selected_cabin_videos:
        items = cabin_video.split('_', 1)
        face_video = 'Face_'+items[1]
        cabin_video_path = os.path.join(cabin_video_dir, cabin_video)
        face_video_path = os.path.join(face_video_dir, face_video)
        selected_starts, selected_ends, all_start_scores, indices_in_cabin_clips = predict_video(cabin_video_path, face_video_path, args)
        video_event_dict = event_dict[cabin_video]
        GT_starts = []
        GT_ends = []
        for k in video_event_dict:
            event_list = video_event_dict[k]
            for event in event_list:
                GT_starts.append(event[0])
                GT_ends.append(event[1])
        GT_starts.sort(reverse=False)
        GT_ends.sort(reverse=False)
        
        if len(selected_starts) != 0:
            relevant_elements = 0
            irrelevant_elements = 0
            for i in range(len(indices_in_cabin_clips)):
                set1 = set(indices_in_cabin_clips[i])
                set2 = set(GT_starts)
                if len(set1.intersection(set2)) != 0:
                    relevant_elements += 1
                else:
                    irrelevant_elements += 1
        
            i = 0
            j = 0
            true_positive = 0
            while i < len(selected_starts):
                while j < len(GT_starts):
                    if GT_starts[j] < selected_starts[i]*4:
                        j += 1
                    elif GT_starts[j] >= selected_starts[i]*4 and GT_starts[j] <= selected_starts[i]*4+15:
                        true_positive += 1
                        break
                    elif GT_starts[j] > selected_starts[i]*4 + 15:
                        break
                i += 1
            false_positive = len(selected_starts)-true_positive
            true_negative = irrelevant_elements-false_positive
            false_negative = relevant_elements - true_positive
            true_positive_rate = float(true_positive)/(true_positive + false_negative + 1e-10)
            false_positive_rate = float(false_positive)/(false_positive + true_negative + 1e-10)
            precision = float(true_positive)/(len(selected_starts) + 1e-10)
            recall = float(true_positive)/(relevant_elements + 1e-10)
            all_true_positive_rate.append(true_positive_rate)
            all_false_positive_rate.append(false_positive_rate)
            all_precision.append(precision)
            all_recall.append(recall)
            print(true_positive_rate, false_positive_rate, precision, recall)
           
        
        
        
        if len(selected_ends) != 0:
            relevant_elements1 = 0
            irrelevant_elements1 = 0
            for i in range(len(indices_in_cabin_clips)):
                set1 = set(indices_in_cabin_clips[i])
                set2 = set(GT_ends)
                if len(set1.intersection(set2)) != 0:
                    relevant_elements1 += 1
                else:
                    irrelevant_elements1 += 1
            i = 0
            j = 0
            true_positive1 = 0
            while i < len(selected_ends):
                while j < len(GT_ends):
                    if GT_ends[j] < selected_ends[i]*4:
                        j += 1
                    elif GT_ends[j] >= selected_ends[i]*4 and GT_ends[j] <= selected_ends[i]*4+15:
                        true_positive1 += 1
                        break
                    elif GT_ends[j] > selected_ends[i]*4 + 15:
                        break
                i += 1
            false_positive1 = len(selected_ends)-true_positive1
            true_negative1 = irrelevant_elements1-false_positive1
            false_negative1 = relevant_elements1-true_positive1
            
            true_positive_rate1 = float(true_positive1)/(true_positive1 + false_negative1 + 1e-10)
            false_positive_rate1 = float(false_positive1)/(false_positive1 + true_negative1 + 1e-10)
            precision1 = float(true_positive1)/(len(selected_ends) + 1e-10)
            recall1 = float(true_positive1)/(relevant_elements1 + 1e-10)
        
            all_true_positive_rate1.append(true_positive_rate1)
            all_false_positive_rate1.append(false_positive_rate1)
            all_precision1.append(precision1)
            all_recall1.append(recall1)
            print(true_positive_rate1, false_positive_rate1, precision, recall1)
            
    average_true_positive_rate  =  sum(all_true_positive_rate)/len(all_true_positive_rate)
    average_false_positive_rate  =  sum(all_false_positive_rate)/len(all_false_positive_rate)
    average_precision  =  sum(all_precision)/len(all_precision)
    average_recall  =  sum(all_recall)/len(all_recall)
    
    average_true_positive_rate1  =  sum(all_true_positive_rate1)/len(all_true_positive_rate1)
    average_false_positive_rate1  =  sum(all_false_positive_rate1)/len(all_false_positive_rate1)
    average_precision1  =  sum(all_precision1)/len(all_precision1)
    average_recall1  =  sum(all_recall1)/len(all_recall1)
    
    print('start_average_true_positive_rate:{0}, start_average_false_positive_rate:{1},  start_average_precision:{2}, start_average_recall:{3}'.format(average_true_positive_rate, average_false_positive_rate, average_precision, average_recall))
    print('end_average_true_positive_rate:{0}, end_average_false_positive_rate:{1},  end_average_precision:{2}, end_average_recall:{3}'.format(average_true_positive_rate1, average_false_positive_rate1, average_precision1, average_recall1))
        
                    
if __name__ == '__main__':
    main()

