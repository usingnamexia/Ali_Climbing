import os
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch

JOINTS=20
class ClimbDataset(Dataset):
    def __init__(self,
                 annot_fname,
                 image_path,
                 sample_processor=None):

        """
        annot_fname = "~/annots/"
        image_path = "~/imgs/"
        """
        super().__init__()
        assert os.path.exists(annot_fname)
        assert os.path.exists(image_path) and os.path.isdir(image_path)
        
        print(">>>lodding from:\n"\
              "   Annot:[{}]\n"\
              "   Image:[{}]".format(annot_fname, image_path))
        
        self.sample_processor = sample_processor
        
        self.mask = torch.ones((1,JOINTS,1),dtype=torch.float32)
        self.mask[:, [14,15,16,17.18,19]] = 0.0
        self.img_path_list = os.listdir(image_path)
        self.img_path = []
        self.annot_path = []
        for i in self.img_path_list:
            self.img_path.append(image_path + i)
            self.annot_path.append(annot_fname + i.split(".")[0] + ".json")

        
    def __len__(self):
        return len(self.img_path_list)
    
    def preprocess_sample_x2ds(self, sample):
        pass
    
    def __getitem__(self,idx):
        sample = {"bbox": None}
        sample["path"] = self.img_path[idx]
        sample["x2ds"] = self.LoadJson(self.annot_path[idx])
        return sample

        
        # xyxy, x2ds, mask = self.preprocess_sample_x2ds(sample)
        
        # if self.sample_processor is not None:
        #     processed_sample = self.sample_processor(cv2.imread(path),
        #                                              x2ds,
        #                                              mask,
        #                                              xyxy=xyxy)


    def LoadJson(self, json_fpath):
        with open(json_fpath,'r') as f:
            json_data = json.load(f)
            N = len(json_data['person'])
            kp2ds = []
            if N == 0:
                return []
            for i in range(N):
                kp2ds.append(json_data['person'][i]['kp'])#pose_keypoints_2d#people
            kp2ds = np.array(kp2ds).reshape((-1,20,3))
        return kp2ds