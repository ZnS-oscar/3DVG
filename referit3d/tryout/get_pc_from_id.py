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
from referit3d.in_out.pt_datasets.utils import mean_rgb_unit_norm_transform
from functools import partial
def sample_scan_object(object, n_points):
    sample = object.sample(n_samples=n_points)
    return np.concatenate([sample['xyz'], sample['color']], axis=1)

if __name__ == '__main__':
    
    
    input_file_path = "for_cls_id_get_obj_id.txt"
    with open(input_file_path, 'r') as file:
        for_cls_id_get_obj_id = json.load(file)

    args = parse_arguments()
    device=torch.device(type='cuda')
    np.random.seed(args.random_seed)
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)
    scans = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    padid=class_to_idx['pad']
    cls_ids=for_cls_id_get_obj_id.keys()

    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
        unit_norm=args.unit_sphere_norm)
    
    cls_id2pcs={}
    for cls_id in cls_ids:
        if cls_id == str(padid):
            continue
        pcs=[]
        obj_ids=for_cls_id_get_obj_id[cls_id]
        for obj_id in obj_ids:
            s_id,o_id=obj_id
            pcs.append(scans[s_id].three_d_objects[o_id])
        pcs=[sample_scan_object(o,args.points_per_object).tolist() for o in pcs]
        pcs=np.array(pcs)
        pcs=np.float32(pcs)
        if object_transformation is not None:
            pcs = object_transformation(pcs)
        # pcs=torch.tensor(pcs).to(device)
        pcs=torch.tensor(pcs)

        cls_id2pcs[int(cls_id)]=pcs
    pcs_padid=np.zeros((len(pcs[0]),len(pcs[0][0])))[np.newaxis,:]
    if object_transformation is not None:
        pcs_padid = object_transformation(pcs_padid)
    pcs_padid=torch.tensor(np.float32(pcs_padid))
    # pcs_padid=torch.tensor(np.float32(pcs_padid)).to(device)
    cls_id2pcs[padid]=pcs_padid
    
    # with open(output_file_path, 'w') as cls_id2pcs_file:
    #     json.dump(cls_id2pcs, cls_id2pcs_file)
    with open("cls_id2pcs.pkl",'wb') as cls_id2pcsfile:
	    pickle.dump(cls_id2pcs, cls_id2pcsfile)

        
        



