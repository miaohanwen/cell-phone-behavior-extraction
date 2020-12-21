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
from collections import defaultdict


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cabin_video_path', type=str, help='path to cabin video')
    parser.add_argument('--face_video_path', type=str, help='path to face video')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--clip_length', default=16, type=int, help='Number of frames in each clip')
    parser.add_argument('--clip_stride', default=4, type=int, help='Number of frames between the starts of two clips')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of clips to process together each time')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    parser.add_argument('--threshold', type=float, help='threshold for start scores and end scores')
    args = parser.parse_args()
    return args


def clip_generation(cabin_video_path, face_video_path, clip_length, clip_stride):
    cabin_frames = os.listdir(cabin_video_path)
    cabin_frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    face_frames = os.listdir(face_video_path)
    face_frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    cabin_frame_length = len(cabin_frames)
    face_frame_length = len(face_frames)
    #     if L2 > L1*5:
    #         cabin_frame_length = L1
    #         face_frame_length = L1*5
    #     else:
    #         cabin_frame_length = L2 // 5
    #         face_frame_length = L2
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


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    args = get_parse()
    cabin_video_path = args.cabin_video_path
    face_video_path = args.face_video_path
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
    
    # rough chunk aggregation
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
            selected_ends.append(end_indice)
    print(selected_starts)
    print(selected_ends)
    selected_start_scores = []
    selected_end_scores = []
    if selected_starts != []:
        for start in selected_starts:
            selected_start_scores.append(all_start_scores[start])
    if selected_ends != []:
        for end in selected_ends:
            selected_end_scores.append(all_end_scores[end])

    # plot
    all_clips = range(len(all_start_scores))
    fig = plt.figure()
    plt.plot(all_clips, all_start_scores, "b.-", label="start scores")
    plt.plot(all_clips, all_end_scores, "r.-", label="end scores")
    if selected_starts != []:
        plt.scatter(selected_starts, selected_start_scores, c='b', marker='*', linewidths=3, label="selected clips including starts")
    if selected_ends != []:
        plt.scatter(selected_ends, selected_end_scores, c='r', marker='*', linewidths=3, label="selected clips including ends")
    plt.legend(loc='upper right')
    plt.ylim(0, 1)
    plt.xlabel("Clip Index")
    plt.ylabel("predicted score")
    plt.show()
    cabin_video_name = os.path.basename(cabin_video_path)
    fig.savefig('figures/plot_{}.png'.format(cabin_video_name))
    
    

if __name__ == '__main__':
    main()