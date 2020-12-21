import pandas as pd
import os
import numpy as np
import argparse
import glob
import re
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters to generate clips')
    parser.add_argument('--cabin_video_dir', help='Cabin View Video directory')
    parser.add_argument('--face_video_dir', help='Face View Video directory')
    parser.add_argument('--cabin_control_video_dir', help='Cabin View Control Video directory')
    parser.add_argument('--face_control_video_dir', help='Face View Control Video directory')
    parser.add_argument('--label_file_path1', help='Path to IVBSS Cell Census Final Event Data - park and no park.xlsx')
    parser.add_argument('--label_file_path2', help='Path to cabin_drivertrips_subset.csv')
    parser.add_argument('--clip_length', type=int, help='Frame length of a clip from cabin view')
    parser.add_argument('--save_path', help='Path to save the preprocessed clips')
    args = parser.parse_args()
    return args


def extract_clips(cabin_video_dir, face_video_dir, video_name, df1, df2, clip_length):
    print(video_name)
    items = video_name.split('_')
    driver_id = int(items[1].lstrip('0'))
    trip_id = int(items[2].lstrip('0'))
    time = int(items[3])
    df3 = df2[(df2['Driver'] == driver_id) & (df2['Trip'] == trip_id)]
    before = df3[df3['VideoTime'] <= time]
    if before.empty:
        first_idx = df3.index[0]
    else:
        first_idx = before.index[-1]
    video_start_cabincount = df3.at[first_idx, 'CabinCount']
    
    cabin_frame_list = os.listdir(os.path.join(cabin_video_dir, video_name))
    cabin_frame_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    video_name1 = 'Face_' + video_name.split('_', 1)[1]
    face_frame_list = os.listdir(os.path.join(face_video_dir, video_name1))
    face_frame_list.sort(key=lambda f: int(re.sub('\D', '', f)))

    df4 = df1[(df1['driver'] == driver_id) & (df1['trip'] == trip_id) & (df1['starttime'] > time) & (
            df1['endtime'] < time + len(cabin_frame_list) / 2 * 100)]
    clip_list = []
    for index, row in df4.iterrows():
        starttime = row['starttime']
        endtime = row['endtime']
        before_start = df3[df3['VideoTime'] <= starttime]
        before_end = df3[df3['VideoTime'] <= endtime]
        if before_start.empty:
            continue
        index1 = before_start.index[-1]
        index2 = before_end.index[-1]
        start_frame_cabincount = df3.at[index1, 'CabinCount']
        end_frame_cabincount = df3.at[index2, 'CabinCount']

        cabin_start_frame = start_frame_cabincount - video_start_cabincount + 1
        cabin_end_frame = end_frame_cabincount - video_start_cabincount + 1

        if row['startid'] != 7:
            start_sample = max(0, cabin_start_frame - clip_length // 2)
            cabin_frames = cabin_frame_list[start_sample:(start_sample + clip_length)]
            face_frames = face_frame_list[start_sample*5:(start_sample + clip_length)*5]
            if len(cabin_frames) == clip_length and len(face_frames) == clip_length*5:
                clip = {
                    'cabin_video_id': video_name,
                    'face_video_id': video_name1,
                    'cabin frames': cabin_frames,
                    'face frames': face_frames,
                    'start': 1,
                    'end': 0,
                    'label': row['startid']
                }
                clip_list.append(clip)

        if row['endid'] != 8:
            if cabin_end_frame + clip_length//2 > len(cabin_frame_list):
                cabin_end_sample = len(cabin_frame_list)
                cabin_frames = cabin_frame_list[(cabin_end_sample - clip_length):cabin_end_sample]
                face_frames = face_frame_list[(len(face_frame_list) - clip_length*5):len(face_frame_list)]
            elif cabin_end_frame - clip_length//2 < 0:
                cabin_frames = cabin_frame_list[0:clip_length]
                face_frames = face_frame_list[0:clip_length*5]
            else:
                cabin_frames = cabin_frame_list[(cabin_end_frame - clip_length//2):(cabin_end_frame - clip_length//2 + clip_length)]   
                face_end_frame = min((cabin_end_frame - clip_length//2 + clip_length)*5, len(face_frame_list))
                face_frames = face_frame_list[(face_end_frame - clip_length*5): face_end_frame]
                
            if len(cabin_frames) == clip_length and len(face_frames) == clip_length*5:
                clip = {
                    'cabin_video_id': video_name,
                    'face_video_id': video_name1,
                    'cabin frames': cabin_frames,
                    'face frames': face_frames,
                    'start': 0,
                    'end': 1,
                    'label': row['startid']
                }
                clip_list.append(clip)
                
            if cabin_end_frame - cabin_start_frame > clip_length:
#                 rand = np.random.randint(cabin_start_frame, cabin_end_frame - clip_length, 1)[0]
                rand_indices = np.random.randint(cabin_start_frame, cabin_end_frame - clip_length, 2)
                rand_starts = []
                for indice in rand_indices:
                    if indice not in rand_starts:
                        rand_starts.append(indice)
                for rand in rand_starts:
                    cabin_frames = cabin_frame_list[rand:(rand + clip_length)]
                    face_end_frame = min((rand + clip_length)*5, len(face_frame_list))
                    face_frames = face_frame_list[(face_end_frame-clip_length*5):face_end_frame]
                    if len(cabin_frames) == clip_length and len(face_frames) == clip_length*5:
                        clip = {
                            'cabin_video_id': video_name,
                            'face_video_id': video_name1,
                            'cabin frames': cabin_frames,
                            'face frames': face_frames,
                            'start': 0,
                            'end': 0,
                            'label': row['startid']
                        }
                        clip_list.append(clip)
#         if cabin_end_frame - cabin_start_frame > clip_length:
#             rand = np.random.randint(cabin_start_frame, cabin_end_frame - clip_length, 1)[0]
#             cabin_frames = cabin_frame_list[rand:(rand + clip_length)]
#             face_end_frame = min((rand + clip_length)*5, len(face_frame_list))
#             face_frames = face_frame_list[(face_end_frame-clip_length*5):face_end_frame]
#             if len(cabin_frames) == clip_length and len(face_frames) == clip_length*5:
#                 clip = {
#                     'cabin_video_id': video_name,
#                     'face_video_id': video_name1,
#                     'cabin frames': cabin_frames,
#                     'face frames': face_frames,
#                     'start': 0,
#                     'end': 0,
#                     'label': row['startid']
#                 }
#                 clip_list.append(clip)

    return clip_list


def extract_control_clips(cabin_control_video_dir, face_control_video_dir, video_name, clip_length):
    cabin_frame_list = os.listdir(os.path.join(cabin_control_video_dir, video_name))
    cabin_frame_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    video_name1 = 'Face_' + video_name.split('_', 1)[1]
    face_frame_list = os.listdir(os.path.join(face_control_video_dir, video_name1))
    face_frame_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    clip_list = []
    if len(cabin_frame_list) > clip_length:
#         rand = np.random.randint(0, len(cabin_frame_list) - clip_length, 1)[0]
        rand_indices = np.random.randint(0, len(cabin_frame_list) - clip_length, 5)
        rand_starts = []
        for indice in rand_indices:
            if indice not in rand_starts:
                rand_starts.append(indice)
        for rand in rand_starts:
            cabin_frames = cabin_frame_list[rand:(rand + clip_length)]
            face_end_frame = min((rand + clip_length)*5, len(face_frame_list))
            face_frames = face_frame_list[(face_end_frame-clip_length*5):face_end_frame]
            if len(cabin_frames) == clip_length and len(face_frames) == clip_length*5:
                clip = {
                    'cabin_video_id': video_name,
                    'face_video_id': video_name1,
                    'cabin frames': cabin_frames,
                    'face frames': face_frames,
                    'start': 0,
                    'end': 0,
                    'label': 0
                }
                clip_list.append(clip)
    return clip_list


def main():
    args = parse_args()
    cabin_video_dir = args.cabin_video_dir
    face_video_dir = args.face_video_dir
    cabin_control_video_dir = args.cabin_control_video_dir
    face_control_video_dir = args.face_control_video_dir
    
    label_file1_path = args.label_file_path1
    label_file2_path = args.label_file_path2
    clip_length = args.clip_length
    save_path = args.save_path
    
    np.random.seed(0)
    
    df1 = pd.read_excel(label_file1_path)
    df2 = pd.read_csv(label_file2_path)

    cabin_video_list = os.listdir(cabin_video_dir)
    cabin_control_video_list = os.listdir(cabin_control_video_dir)

    driver_ids = []
    for video_name in cabin_video_list:
        items = video_name.split('_')
        driver_id = items[1]
        if driver_id not in driver_ids:
            driver_ids.append(driver_id)
    L = len(driver_ids)
    indices = np.random.permutation(L)
    num_train_driver_ids = int(L * 0.7)
    num_val_driver_ids = int(L * 0.2)
    num_test_driver_ids = L - num_train_driver_ids - num_val_driver_ids
    train_driver_ids = [driver_ids[i] for i in indices[:num_train_driver_ids]]
    val_driver_ids = [driver_ids[i] for i in indices[num_train_driver_ids:(num_train_driver_ids + num_val_driver_ids)]]
    test_driver_ids = [driver_ids[i] for i in indices[(num_train_driver_ids + num_val_driver_ids):]]
    train_video_list = []
    val_video_list = []
    test_video_list = []
    for video_name in cabin_video_list:
        if video_name.split('_')[1] in train_driver_ids:
            train_video_list.append(video_name)
        elif video_name.split('_')[1] in val_driver_ids:
            val_video_list.append(video_name)
        elif video_name.split('_')[1] in test_driver_ids:
            test_video_list.append(video_name)

    train_control_video_list = []
    val_control_video_list = []
    test_control_video_list = []
    for video_name in cabin_control_video_list:
        if video_name.split('_')[1] in train_driver_ids:
            train_control_video_list.append(video_name)
        elif video_name.split('_')[1] in val_driver_ids:
            val_control_video_list.append(video_name)
        elif video_name.split('_')[1] in test_driver_ids:
            test_control_video_list.append(video_name)

    train_video_clips = []
    for video_name in train_video_list:
        clips = extract_clips(cabin_video_dir, face_video_dir, video_name, df1, df2, clip_length)
        train_video_clips += clips

    train_control_video_clips = []
    for video_name in train_control_video_list:
        clip = extract_control_clips(cabin_control_video_dir, face_control_video_dir, video_name, clip_length)
        train_control_video_clips += clip

    val_video_clips = []
    for video_name in val_video_list:
        clips = extract_clips(cabin_video_dir, face_video_dir, video_name, df1, df2, clip_length)
        val_video_clips += clips

    val_control_video_clips = []
    for video_name in val_control_video_list:
        clip = extract_control_clips(cabin_control_video_dir, face_control_video_dir, video_name, clip_length)
        val_control_video_clips += clip

    test_video_clips = []
    for video_name in test_video_list:
        clips = extract_clips(cabin_video_dir, face_video_dir, video_name, df1, df2, clip_length)
        test_video_clips += clips

    test_control_video_clips = []
    for video_name in test_control_video_list:
        clip = extract_control_clips(cabin_control_video_dir, face_control_video_dir, video_name, clip_length)
        test_control_video_clips += clip

    video_clips = {
        'train_video_clips': train_video_clips,
        'train_control_video_clips': train_control_video_clips,
        'val_video_clips': val_video_clips,
        'val_control_video_clips': val_control_video_clips,
        'test_video_clips': test_video_clips,
        'test_control_video_clips': test_control_video_clips,
        'train_video_list': train_video_list,
        'val_video_list': val_video_list,
        'test_video_list': test_video_list,
        'train_control_video_list': train_control_video_list,
        'val_control_video_list': val_control_video_list,
        'test_control_video_list': test_control_video_list
    }

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'video_clips.json'), 'w') as f:
        f.write(json.dumps(video_clips, indent=4))


if __name__ == "__main__":
    main()