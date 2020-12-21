import pandas as pd
import os
import numpy as np
import argparse
import glob
import re
import json
import random
import pickle
from collections import defaultdict


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cabin_video_dir', type=str, help='directory to cabin video')
    parser.add_argument('--label_file_path1', help='Path to IVBSS Cell Census Final Event Data - park and no park.xlsx')
    parser.add_argument('--label_file_path2', help='Path to cabin_drivertrips_subset.csv')
    parser.add_argument('--video_clip_file', help='extract the test video list')
    parser.add_argument('--save_path')
    args = parser.parse_args()
    return args


def extract_events(cabin_video_dir, cabin_video_name, df1, driver_id, trip_id, time, df3, video_start_cabincount):
    frame_list = os.listdir(os.path.join(cabin_video_dir, cabin_video_name))
    frame_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    df1 = df1[(df1['driver'] == driver_id) & (df1['trip'] == trip_id) & (df1['starttime'] > time) & (df1['endtime'] < time + len(frame_list)/2*100)]
    
#     start_frame_list = []
#     end_frame_list = []
    events = defaultdict(list)
    for index, row in df1.iterrows():
        starttime = row['starttime']
        endtime = row['endtime']
        before_start = df3[df3['VideoTime'] <= starttime]
        before_end = df3[df3['VideoTime'] <= endtime]
        if before_start.empty:
            index1 = df3.index[0]
        else:
            index1 = before_start.index[-1]
        index2 = before_end.index[-1]
        start_frame_cabincount = df3.at[index1, 'CabinCount']
        start_frame = max(start_frame_cabincount - video_start_cabincount + 1, 1)
        end_frame_cabincount = df3.at[index2, 'CabinCount']
        end_frame = min(end_frame_cabincount - video_start_cabincount + 1, len(frame_list))
        startid = row['startid']
        if startid == 1 or startid == 3:
            event = 1
        elif startid == 5 or startid == 7:
            event = 2
        events[event].append([start_frame, end_frame])
           
#         if df1.at[index, 'startid'] != 7:
#             start_frame_list.append(start_frame_cabincount - video_start_cabincount + 1)
#         if df1.at[index, 'endid'] != 8:
#             end_frame_list.append(min(end_frame_cabincount - video_start_cabincount + 1, len(frame_list)))
    return events
     
        
def main():
    args = get_parse()
    cabin_video_dir = args.cabin_video_dir
    label_file_path1 = args.label_file_path1
    label_file_path2 = args.label_file_path2
    video_clip_file = args.video_clip_file
    save_path = args.save_path
    
    with open(video_clip_file, 'r') as f:
        video_clip_info = json.load(f)
    test_video_list = video_clip_info['test_video_list']

    
    df1 = pd.read_excel(label_file_path1)
    df2 = pd.read_csv(label_file_path2)

    key_frame_dict = dict()
    for cabin_video_name in test_video_list:
        items = cabin_video_name.split('_')
        driver_id = int(items[1].lstrip('0'))
        trip_id = int(items[2].lstrip('0'))
        time = int(items[3])
        df3 = df2[(df2['Driver'] == driver_id) & (df2['Trip'] == trip_id)]
        before = df3[df3['VideoTime'] <= time]
        if before.empty:
            first_idx = df3.index[0]
        else:
            first_idx = before.index[-1]
        video_start_cabincount = df3.at[first_idx,'CabinCount']
        print(video_start_cabincount)
#         start_frame_list, end_frame_list = extract_key_frames(cabin_video_dir, cabin_video_name, df1, driver_id, trip_id, time, df3, video_start_cabincount)
        events = extract_events(cabin_video_dir, cabin_video_name, df1, driver_id, trip_id, time, df3, video_start_cabincount)
        print(events)
        key_frame_dict[cabin_video_name] = events
    
    with open(save_path, 'wb') as f:
        pickle.dump(key_frame_dict, f)
        
        

if __name__ == "__main__":
    main()