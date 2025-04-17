import math
import os
import argparse
import random
import cv2
import json
from flow_functions import check_and_create_path
import numpy as np
from all_opt import change_opts
from params import params

def flowtext_based(args):
    from flowtext_method import flowtext_based_gen,model_init
    base_dir = args.video_dir
    save_dir = args.save_dir
    check_and_create_path(save_dir)
    videos = os.listdir(base_dir)
    videos = sorted(os.listdir(base_dir))
    flow_model, depth_model, seg_model = model_init()
    for video in videos:
        if os.path.exists(os.path.join(args.save_dir, video[:-4])):
            print(f'{video} output already exists')
            continue
        cap = cv2.VideoCapture(os.path.join(base_dir,video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频帧数
        itv = math.ceil(frame_count / args.frame_len)
        for i in range(itv):
            rg = [i,frame_count,itv]
            print(f'process {video} on itv:{itv}')
            flowtext_based_gen(video_path=os.path.join(base_dir,video), rg=rg, save=os.path.join(save_dir,video[:-4]+f'_{i}'),flow_model = flow_model,
                depth_model = depth_model,seg_model = seg_model)

def codef_based(args):
    from codef_method import codef_based_gen
    from Segment_and_Track_Anything.SegTracker import SegTracker
    from Segment_and_Track_Anything.model_args import aot_args ,sam_args ,segtracker_args
    from flow_estimate import new_flow_init
    base_dir = args.video_dir
    save_dir = args.save_dir
    check_and_create_path(save_dir)
    videos = os.listdir(base_dir)
    videos = sorted(os.listdir(base_dir))
    config = json.load(open(args.config))
    segtracker = SegTracker(params['segtracker']['segtracker_args'], params['segtracker']['sam_args'], params['segtracker']['aot_args'])
    fl_model = new_flow_init()
    change_opts(config,args)
    for video in videos:
        if os.path.exists(os.path.join(args.save_dir, video[:-4])):
            print(f'{video} output already exists')
            continue
        print(f'process {video}')
        #video_path = os.path.join(videos_dir, video)
        codef_based_gen(video,fl_model,segtracker,args)

    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='flowtext-based',choices=['flowtext-based','codef-based'],help='background video')
    parser.add_argument('--video_dir', default='/home/yl/3nP5KcPsPDg.mp4', help='background video')
    parser.add_argument('--frame_len', default=40, help='range of the video')
    parser.add_argument('--save_dir', default='/home/yl/temp_video/3nP5KcPsPDg_10018', help='save path')
    parser.add_argument('--seed', default='10018', help='random seed')
    parser.add_argument('--config', default='codef_config.json', help='CoDeF config')
    args = parser.parse_args()
    seed = int(args.seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.method=='flowtext-based':
        flowtext_based(args)
    else:
        codef_based(args)
