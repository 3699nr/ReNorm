# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from scipy.io import loadmat
from glob import glob
import random
# from fastreid.data.datasets import DATASET_REGISTRY
# from fastreid.data.datasets.bases import ImageDataset
import pdb

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

# __all__ = ['DG_PRID', ]


@DATASET_REGISTRY.register()
class PRID(ImageDataset):

    def __init__(self, root, split_id=0, **kwargs):
        # self.dataset_dir = "datasets/PRID2011"
        self._junk_pids = list(range(201, 750))

        self.dataset_dir = root


        self.cam_a_dir = os.path.join(
            self.dataset_dir, 'PRID', 'single_shot', 'cam_a'
        )
        self.cam_b_dir = os.path.join(
            self.dataset_dir, 'PRID', 'single_shot', 'cam_b'
        )
        self.split_path = os.path.join(self.dataset_dir, 'PRID', 'splits_single_shot.json')

        self.prepare_split()
        splits = self.read_json(self.split_path)
        split = splits[split_id]

        query, gallery = self.process_split(split)
        train = []

        super(PRID, self).__init__(train, query, gallery, **kwargs)
        

    def prepare_split(self):
        if not os.path.exists(self.split_path):
            print('Creating splits ...')

            splits = []
            for _ in range(10):
                # randomly sample 100 IDs for train and use the rest 100 IDs for test
                # (note: there are only 200 IDs appearing in both views)
                pids = [i for i in range(1, 201)]
                train_pids = random.sample(pids, 100)
                train_pids.sort()
                test_pids = [i for i in pids if i not in train_pids]
                split = {'train': train_pids, 'test': test_pids}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            self.write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    def process_split(self, split):
        # train_pids = split['train']
        test_pids = split['test']


        # query and gallery
        query, gallery = [], []
        for pid in test_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_a_path = os.path.join(self.cam_a_dir, img_name)
            query.append((img_a_path, pid, 0))
            img_b_path = os.path.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
            
        for pid in range(201, 750):
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_b_path = os.path.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
            


        return query, gallery

    def read_json(self, fpath):
        import json
        """Reads json file from a path."""
        with open(fpath, 'r') as f:
            obj = json.load(f)
        return obj

    def write_json(self, obj, fpath):
        import json
        """Writes to a json file."""
        self.mkdir_if_missing(os.path.dirname(fpath))
        with open(fpath, 'w') as f:
            json.dump(obj, f, indent=4, separators=(',', ': '))

    def mkdir_if_missing(self, dirname):
        import errno
        """Creates dirname if it is missing."""
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

# PRID('/home/dj/reid/data/PRID')