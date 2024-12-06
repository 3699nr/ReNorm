'''
Description: 
Author: murphys_new_law_dj
Date: 2023-01-13 09:48:09
'''

import numpy as np
import pdb
from glob import glob
import re
import os


"""
id: 1467,  images: 14096(labeled)  [detected 14097 ]
"""

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CUHK03All(ImageDataset):
    
    dataset_dir = 'cuhk03-np'
    dataset_name = "CUHK03All"  
    
    def __init__(self, root, split_id=0, cuhk03_labeled=True, cuhk03_classic_split=False, **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        self.images_dir = os.path.join(self.dataset_dir, 'labeled') ## labeled + detected
        # self.images_dir_1 = os.path.join(root, 'detected')

        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
#        self.camstyle_path = 'bounding_box_train_camstyle'
        # self.train, self.query, self.gallery = [], [], []
        self.all_pids_dic = {}
        self.datasets = []
        # self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.load()

        train = self.datasets

        super(CUHK03All, self).__init__(train, [], [], **kwargs)   

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        # all_pids = {}
        # all_pids = []
        # ret = []
        fpaths = sorted(glob(os.path.join(path, '*.png'))) # + glob(os.path.join(self.images_dir_1, path, '*.png')))
        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in self.all_pids_dic:
                    self.all_pids_dic[pid] = len(self.all_pids_dic)
            # else:
            # if pid not in self.all_pids:
                # all_pids[pid] = pid
                # self.all_pids.append(pid)
            pid = self.all_pids_dic[pid]
            cam -= 1
            self.datasets.append((os.path.join(path, fname), self.dataset_name + "_" + str(pid), self.dataset_name + "_" + str(cam)))
        # return self.ret  # , int(len(self.all_pids))

    def load(self):
        self.preprocess(os.path.join(self.images_dir, self.train_path))
        self.preprocess(os.path.join(self.images_dir, self.gallery_path))
        self.preprocess(os.path.join(self.images_dir, self.query_path))

        self.all_pids = len(self.all_pids_dic)

        # print(self.__class__.__name__, "dataset loaded")
        # print("  subset   | # ids | # images")
        # print("  ---------------------------")
        # print("  train    | {:5d} | {:8d}"
        #       .format(self.all_pids, len(self.datasets)))
        # print("  query    | {:5d} | {:8d}"
        #       .format(self.num_query_pids, len(self.query)))
        # print("  gallery  | {:5d} | {:8d}"
        #       .format(self.num_gallery_pids, len(self.gallery)))

# CUHK03_all('/home/ubuntu/data/cuhk03')