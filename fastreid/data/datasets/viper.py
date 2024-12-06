# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

# from fastreid.data.datasets import DATASET_REGISTRY
# from fastreid.data.datasets.bases import ImageDataset

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

# __all__ = ['VIPeR', ]



@DATASET_REGISTRY.register()
class VIPeR(ImageDataset):

    def __init__(self, root, split_id=0,  **kwargs):
        # if isinstance(root, list):
        #     type = root[1]
        #     self.root = root[0]
        # else:
        self.root = root
        self.type_list = ['split_1a', 'split_2a', 'split_3a', 'split_4a', 'split_5a', 'split_6a', 'split_7a', 'split_8a', \
                            'split_9a', 'split_10a']   

        # self.dataset_dir = "datasets/VIPeR"

        self.train_dir = os.path.join(self.root, 'VIPeR', self.type_list[split_id], 'train')
        self.query_dir = os.path.join(self.root, 'VIPeR', self.type_list[split_id], 'query')
        self.gallery_dir = os.path.join(self.root, 'VIPeR', self.type_list[split_id], 'gallery')

        # required_files = [
        #     self.train_dir,
        #     self.query_dir,
        #     self.gallery_dir,
        # ]
        # self.check_before_run(required_files)

        # train = self.process_train(self.train_dir, is_train = True)
        query, self.all_ids = self.process_train(self.query_dir)
        gallery, _ = self.process_train(self.gallery_dir)
        train = []

        # super().__init__(train, query, gallery, **kwargs)

        # print(self.__class__.__name__, "dataset loaded")
        # print("  subset   | # ids | # images")
        # print("  ---------------------------")
        # # print("  train    | {:5d} | {:8d}"
        # #       .format(len(self.train_1), len(self.)))
        # print("  query    | {:5d} | {:8d}"
        #       .format(len(self.q_ids), len(self.query)))
        # print("  gallery  | {:5d} | {:8d}"
        #       .format(len(self.g_ids), len(self.gallery))) 
        super(VIPeR, self).__init__(train, query, gallery, **kwargs)

    def process_train(self, path):
        data = []
        pids = []
        img_list = glob(os.path.join(path, '*.png'))
        for img_path in img_list:
            img_name = img_path.split('/')[-1] # p000_c1_d045.png
            split_name = img_name.split('_')
            pid = int(split_name[0][1:])
            if pid not in pids:
                pids.append(pid)
            camid = int(split_name[1][1:])
            # dirid = int(split_name[2][1:-4])
            data.append((img_path, pid, camid))

        return data, pids

# VIPeR('/home/ubuntu/data/viper')