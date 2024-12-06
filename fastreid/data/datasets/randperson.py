import os.path as osp
import random

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from glob import glob

@DATASET_REGISTRY.register()
class RandPerson(ImageDataset):

    dataset_dir = "/home/nr_2022/data/randperson_subset/randperson_subset/"
    dataset_name = "randperson"
    def __init__(self, root='datasets', **kwargs):
        
        self.images_dir = '' 
        self.img_path = "/home/nr_2022/data/randperson_subset/randperson_subset/"
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        train = self.preprocess(self.train_path)
        super().__init__(train, [], [], **kwargs)
    def preprocess(self, train_path):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*g')))
        data = []
        all_pids = {}

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            fields = fname.split('_')
            #pid = int(fields[0])
            pid = self.dataset_name + "_" + fields[0]
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  # relabel
            camid = int(fields[2][1:])  # make it starting from 0 #fix bug -1
            #data.append((self.img_path + '/' + fname, pid, camid))
            data.append([fpath, pid, camid])
        return data


@DATASET_REGISTRY.register()
class RandPerson_1(ImageDataset):

    dataset_dir = "/home/nr_2022/data/randperson_subset/randperson_subset/"
    dataset_name = "randperson"
    def __init__(self, root='datasets', **kwargs):
        
        self.images_dir = '' 
        self.img_path = "/home/nr_2022/data/randperson_subset/randperson_subset/"
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        train = self.preprocess(self.train_path)
        super().__init__(train, [], [], **kwargs)
    def preprocess(self, train_path):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*g')))
        data = []
        all_pids = {}

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            fields = fname.split('_')
            #pid = int(fields[0])
            pid = self.dataset_name + "_" + fields[0]
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  # relabel
            camid = int(fields[2][1:])  # make it starting from 0 #fix bug -1
            if camid == 0 or camid == 1:

            #data.append((self.img_path + '/' + fname, pid, camid))
                data.append([fpath, pid, camid])
        return data


@DATASET_REGISTRY.register()
class RandPerson_2(ImageDataset):

    dataset_dir = "/home/nr_2022/data/randperson_subset/randperson_subset/"
    dataset_name = "randperson"
    def __init__(self, root='datasets', **kwargs):
        
        self.images_dir = '' 
        self.img_path = "/home/nr_2022/data/randperson_subset/randperson_subset/"
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        train = self.preprocess(self.train_path)
        super().__init__(train, [], [], **kwargs)
    def preprocess(self, train_path):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*g')))
        data = []
        all_pids = {}

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            fields = fname.split('_')
            #pid = int(fields[0])
            pid = self.dataset_name + "_" + fields[0]
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  # relabel
            camid = int(fields[2][1:])  # make it starting from 0 #fix bug -1
            if camid == 2 or camid == 3 or camid == 1:

            #data.append((self.img_path + '/' + fname, pid, camid))
                data.append([fpath, pid, camid])
        return data
    
@DATASET_REGISTRY.register()
class RandPerson_3(ImageDataset):

    dataset_dir = "/home/nr_2022/data/randperson_subset/randperson_subset/"
    dataset_name = "randperson"
    def __init__(self, root='datasets', **kwargs):
        
        self.images_dir = '' 
        self.img_path = "/home/nr_2022/data/randperson_subset/randperson_subset/"
        self.train_path = self.img_path
        self.gallery_path = ''
        self.query_path = ''
        self.train = []
        self.gallery = []
        self.query = []
        train = self.preprocess(self.train_path)
        super().__init__(train, [], [], **kwargs)
    def preprocess(self, train_path):
        fpaths = sorted(glob(osp.join(self.images_dir, self.train_path, '*g')))
        data = []
        all_pids = {}

        for fpath in fpaths:
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            fields = fname.split('_')
            #pid = int(fields[0])
            pid = self.dataset_name + "_" + fields[0]
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  # relabel
            camid = int(fields[2][1:])  # make it starting from 0 #fix bug -1
            if camid == 1 or camid == 3:

            #data.append((self.img_path + '/' + fname, pid, camid))
                data.append([fpath, pid, camid])
        return data