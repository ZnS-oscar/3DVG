

import sys
import json
import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored
import numpy as np

import pickle
from functools import partial
from referit3d.in_out.pt_datasets.utils import instance_labels_of_context
from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.utils.evaluation import AverageMeter
from referit3d.utils.tf_visualizer import Visualizer
from referit3d.models.referit3d_net import ReferIt3DNet_transformer
from referit3d.models.referit3d_net_utils import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions
from transformers import BertTokenizer, BertModel
from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data
from referit3d.models.referit3d_net_utils import cls_pred_stats
import pickle
from referit3d.in_out.pt_datasets.utils import instance_labels_of_context
from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.utils.tf_visualizer import Visualizer
from referit3d.models.referit3d_net import ReferIt3DNet_transformer
from referit3d.models.referit3d_net_utils import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions
from transformers import BertTokenizer, BertModel
from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data
def sample_pc(pc):
    idx = np.random.choice(1024, 1024, replace=1024 < 1024)
    return pc[idx]
def mean_color():
    return np.zeros((1, 3),dtype=np.float32)
def get_obj_pcs():
    with open('chairs.pkl', 'rb') as chairs_file:
        pcs = pickle.load(chairs_file)
    pcs = pcs[1:]
    b, n, _ = pcs.shape
    z = np.zeros((b, n, 3))
    ret = np.concatenate((pcs, z), axis=-1)
    return ret
def spin_pts(pts):
    mat = torch.tensor([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]],dtype=torch.float64)

    rotated_pc = torch.matmul(mat,pts[:,:3].T)
    rotated_pc=rotated_pc.T
    return torch.concat((rotated_pc,pts[:,3:]),dim=-1)


def mean_rgb_unit_norm_transform(segmented_objects, mean_rgb, unit_norm, epsilon_dist=10e-6, inplace=True):
    """
    :param segmented_objects: K x n_points x 6, K point-clouds with color.
    :param mean_rgb:
    :param unit_norm:
    :param epsilon_dist: if max-dist is less than this, we apply not scaling in unit-sphere.
    :param inplace: it False, the transformation is applied in a copy of the segmented_objects.
    :return:
    """
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    segmented_objects[:, :, 3:6] -= np.expand_dims(mean_rgb, 0)

    # center xyz
    if unit_norm:
        xyz = segmented_objects[:, :, :3]
        mean_center = xyz.mean(axis=1)
        xyz -= np.expand_dims(mean_center, 1)
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)), -1)
        max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        segmented_objects[:, :, :3] = xyz

    return segmented_objects

if __name__ == '__main__':
    device = torch.device('cuda')
    cls_id2pcs_path = "cls_id2pcs.pkl"
    with open(cls_id2pcs_path, 'rb') as cls_id2pcsfile:
        cls_id2pcs = pickle.load(cls_id2pcsfile)
    chair=get_obj_pcs()[1]
    
    pcs=np.array(spin_pts(torch.tensor(chair)) )
    sampled_pcs = sample_pc(pcs)
    sampled_pcs=np.array(sampled_pcs)
    mean_rgb=mean_color()
    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
        unit_norm=True)
    reg_pcs = object_transformation(sampled_pcs[np.newaxis,...])

    reg_pcs = torch.tensor(reg_pcs).float()
    cls_id2pcs[87]=reg_pcs

        
    with open("cls_id2pcschair.pkl",'wb') as reg_pcsfile:
        pickle.dump(cls_id2pcs, reg_pcsfile)
        