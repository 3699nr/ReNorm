# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
import glob
import copy
import random
from collections import defaultdict
# from fastreid.data.datasets import DATASET_REGISTRY
# from fastreid.data.datasets.bases import ImageDataset

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

# __all__ = ['DG_iLIDS', ]

"""
119个id,共476张图片.前4个数字是id.
DG: 随机选r个id作为query,这r个id的剩余图片组成gallery.同样重复10次取平均结果.参考DIMN, r=60
"""
@DATASET_REGISTRY.register()
class iLIDS(ImageDataset):

    def __init__(self, root, split_id = 0, **kwargs):
        # self.dataset_dir = "datasets/iLIDS"
        self.root = root
        self.dataset_dir = self.root
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        self.data_dir = os.path.join(self.dataset_dir, 'iLIDS', 'images')
        self.split_path = os.path.join(self.dataset_dir, 'iLIDS', 'splits.json')

        # required_files = [self.dataset_dir, self.data_dir]
        # self.check_before_run(required_files)

        self.prepare_split()
        splits = self.read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but '
                'expected between 0 and {}'.format(split_id,
                                                   len(splits) - 1)
            )
        split = splits[split_id]

        query, gallery = self.process_split(split)
        train = []

        super(iLIDS, self).__init__(train, query, gallery, **kwargs)

        # print(self.__class__.__name__, "dataset loaded")
        # print("  subset   | # ids | # images")
        # print("  ---------------------------")
        # # print("  train    | {:5d} | {:8d}"
        # #       .format(len(self.train_1), len(self.)))
        # print("  query    | {:5d} | {:8d}"
        #       .format(len(self.query), len(self.query)))
        # print("  gallery  | {:5d} | {:8d}"
        #       .format(len(self.gallery), len(self.gallery))) 


    def prepare_split(self):
        if not os.path.exists(self.split_path):
            print('Creating splits ...')

            paths = glob.glob(os.path.join(self.data_dir, '*.jpg'))
            img_names = [os.path.basename(path) for path in paths]
            num_imgs = len(img_names)
            assert num_imgs == 476, 'There should be 476 images, but ' \
                                    'got {}, please check the data'.format(num_imgs)

            # store image names
            # image naming format:
            #   the first four digits denote the person ID
            #   the last four digits denote the sequence index
            pid_dict = defaultdict(list)
            for img_name in img_names:
                pid = int(img_name[:4])
                pid_dict[pid].append(img_name)
            pids = list(pid_dict.keys())
            num_pids = len(pids)
            assert num_pids == 119, 'There should be 119 identities, ' \
                                    'but got {}, please check the data'.format(num_pids)

            num_train_pids = int(num_pids * 0.5)

            splits = []
            for _ in range(10):
                # randomly choose num_train_pids train IDs and the rest for test IDs
                pids_copy = copy.deepcopy(pids)
                random.shuffle(pids_copy)
                train_pids = pids_copy[:num_train_pids]
                test_pids = pids_copy[num_train_pids:]

                train = []
                query = []
                gallery = []

                # for train IDs, all images are used in the train set.
                for pid in train_pids:
                    img_names = pid_dict[pid]
                    train.extend(img_names)

                # for each test ID, randomly choose two images, one for
                # query and the other one for gallery.
                for pid in test_pids:
                    img_names = pid_dict[pid]
                    samples = random.sample(img_names, 2)
                    query.append(samples[0])
                    gallery.append(samples[1])

                split = {'train': train, 'query': query, 'gallery': gallery}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            self.write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    # def get_pid2label(self, img_names):
    #     pid_container = set()
    #     for img_name in img_names:
    #         pid = int(img_name[:4])
    #         pid_container.add(pid)
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}
    #     return pid2label

    def parse_img_names(self, img_names, pid2label=None):
        data = []

        for img_name in img_names:
            pid = int(img_name[:4])
            if pid2label is not None:
                pid = pid2label[pid]
            camid = int(img_name[4:7]) - 1 # 0-based
            img_path = os.path.join(self.data_dir, img_name)
            data.append((img_path, pid, camid))

        return data

    def process_split(self, split):
        # train_pid2label = self.get_pid2label(split['train'])
        # train = self.parse_img_names(split['train'], train_pid2label)
        query = self.parse_img_names(split['query'])
        gallery = self.parse_img_names(split['gallery'])

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

# iLIDS('/home/ubuntu/data/iLIDS')