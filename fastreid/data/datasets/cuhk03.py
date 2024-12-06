# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import json
import os.path as osp

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils.file_io import PathManager
from .bases import ImageDataset

from glob import glob
import re
import os


@DATASET_REGISTRY.register()
class CUHK03(ImageDataset):
    """CUHK03.

    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!>`_

    Dataset statistics:
        - identities: 1360.
        - images: 13164.
        - cameras: 6.
        - splits: 20 (classic).
    """
    dataset_dir = 'cuhk03-np'
    # dataset_url = None
    dataset_name = "cuhk03"

    def __init__(self, root='datasets', split_id=0, cuhk03_labeled=True, cuhk03_classic_split=False, **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir, "labeled")


        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
#        self.camstyle_path = 'bounding_box_train_camstyle'
        train, query, gallery = [], [], []
        # self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        train = self.preprocess(osp.join(self.dataset_dir, self.train_path))  
        gallery = self.preprocess(osp.join(self.dataset_dir, self.gallery_path), False)
        query = self.preprocess(osp.join(self.dataset_dir, self.query_path) , False)

        super(CUHK03, self).__init__(train, query, gallery, **kwargs)


    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        ret = []
        fpaths = sorted(glob(os.path.join(path, '*.png')))

        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid, camid = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}


        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid, camid = map(int, pattern.search(fname).groups())
            camid -= 1

            if relabel:
                pid = pid2label[pid]
                ret.append((os.path.join(path, fname), self.dataset_name + "_" + str(pid), self.dataset_name + "_" + str(camid)))
            else:
                ret.append((os.path.join(path, fname), pid, camid))

        return ret # int(len(all_pids))
    