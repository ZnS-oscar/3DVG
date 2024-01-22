import numpy as np
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers
import torch
import pickle

class AlterScannetPc:
    def __init__(self,cls_id2pcsfile_path):
        self.cls_id2pcsfile_path=cls_id2pcsfile_path
        with open(cls_id2pcsfile_path, 'rb') as cls_id2pcsfile:
            self.cls_id2pcs = pickle.load(cls_id2pcsfile)

    def get_scannet_pc(self,clsid,npoint):
        ret_val=self.cls_id2pcs[clsid][0]
        return self.cls_id2pcs[clsid][0]