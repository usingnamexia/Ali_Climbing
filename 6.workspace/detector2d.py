# 输入参数 视频路径
import sys
import os
import time
import cv2
import numpy as np
import torch

sys.path.append("../0.mc_utils/")

from detector.yolov4_config import YOLO_CFG
from detector.yolov4_wrapper import YoloV4Detector
from common.one_eurio_filter import OneEuroFilter
from common.unity_visualizer import draw_kp2d_to_image
from common.tools_cv import draw_kp
from syp_hmr_e2e.x2ds_config import X2DS_CPN50_CFG
from syp_hmr_e2e.x2ds_cpn50_wrapper import X2dsCPN50Wrapper


WIDTH=56
HEIGHT=56
device= "cuda:0"

x2ds_filter = OneEuroFilter(min_cutoff=0.05, beta=0.03,d_cutoff=0.02)

cfg = YOLO_CFG.clone()
cfg.confidence_threshold=0.6 # 0.8
cfg.device = device
cfg.framesize = 320
cfg.iou_threshold = 0.6
print(cfg)
human_detector = YoloV4Detector()


## exp-resnet ##
x2ds_cfg = X2DS_CPN50_CFG.clone()
x2ds_cfg.device = device
x2ds_cfg.basic.xyxy_conf_thresh = 0.1  # 0.5 bbox置信度在这个阈值下时，2d点才会输出
x2ds_cfg.basic.crop_scale = (1.5,1.1)
x2ds_cfg.conf_reader.sigma = 7.0       # 搜索半径，可以改大3~16
x2ds_cfg.model = "../12.models/hmr_e2e/20200904-cpn50-224x224j20-climb-0.5507786451159297.pth" # retrained
# x2ds_cfg.model = "../12.models/hmr_e2e/20200817-cpn50-224x224j20-2.pth"
x2ds_cfg.network.hmap_channels = 20
print(x2ds_cfg)
x2ds_resnet50 = X2dsCPN50Wrapper(x2ds_cfg)

def detector2d(video_fname):
    ret = dict()
    
    cap = cv2.VideoCapture(video_fname)

    # cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


#     window_name = "yolov5-dsnt2d"
#     cv2.namedWindow(window_name)
    frame_count = 0
    
    framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
    frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

    # save
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fps = cap.get(cv2.CAP_PROP_FPS)
    #size = (framewidth*2, frameheight*2)

    # 保存视频
    #videoname = video_fname.split("/")[-1]
    #out = cv2.VideoWriter('D:/XiaHaiPeng/output/WIN_20200921_11_31_00_Pro/WIN_20200921_11_31_00_Pro_kp2d.mp4', fourcc, fps, size)

    #im_number = 0
    while True:
        is_valid,frame = cap.read()
        if not is_valid:
            break
        pred = []
        ##step1 preprocess image
        raw_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #HxWxC

        ##step2.human detector
        det_ret = human_detector(raw_image)
        if det_ret is not None:
            batch_xyxy = det_ret["xyxy"]
            batch_conf = det_ret["conf"]
            
            bigger_conf = -1
            for i,v in enumerate(batch_conf):
                if v > bigger_conf:
                    idx = i
                    bigger_conf = v
                    
            ##step2.prepare to 2D image
            raw_xyxy = batch_xyxy[idx]
            
            
            if batch_conf.shape[0] > 1:
                batch_conf = batch_conf[idx].unsqueeze(0)
                batch_xyxy = batch_xyxy[idx].unsqueeze(0)
            #print(batch_xyxy.shape, batch_conf.shape)
#                 cv2.rectangle(frame,(raw_xyxy[0],raw_xyxy[1]),(raw_xyxy[2],raw_xyxy[3]),(0,255,0),2)
#                 cv2.putText(frame, 
#                             "P:{:.3}".format(batch_conf[i]), (raw_xyxy[0],raw_xyxy[1]), 
#                              cv2.FONT_HERSHEY_SIMPLEX, 
#                              0.6, 
#                              (0,255,0), 
#                              1,
#                              cv2.LINE_AA)
            
            ##step3.2D pose to heatmap
            pred = x2ds_resnet50(raw_image,batch_xyxy,batch_conf)
            if pred is not None:
                pred_x2ds = pred[...,:2]
                pred_conf = pred[...,-1:]
                conf_x2ds = pred_x2ds*(pred_conf>0.1)

                ##step6.visualize x2ds
                frame = draw_kp(frame,conf_x2ds,fmt="mc16",color=(0,255,0))
                for i in range(len(pred_x2ds)):
                    draw_kp2d_to_image(conf_x2ds[i],frame,radius=4,font_scale=0.6,color=(255,0,0)) # open when draw
        cv2.imshow('kp2d',cv2.resize(frame,(960,540)))
        cv2.waitKey(1)
        pose = map_mc14_to_op(pred)
        #print(pose)
        if pose != None:
            ret[frame_count] = pose.numpy()
        frame_count += 1
        # 保存视频
        #out.write(frame)
        
    cv2.destroyWindow('kp2d')
    cap.release()
    return ret

def map_mc14_to_op(x2ds):
    if not len(x2ds):
        return None
    select_index = [24,23,22,21,20,9,8,10,7,11,6,3,2,4,1,5,0,12,13,14,15,16,17,18,19]
    x2ds = torch.cat((x2ds.cpu(), torch.zeros(1,5,3)),axis=1)
    x2ds = x2ds[:,select_index,:]
    return x2ds
    