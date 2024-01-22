import open3d as o3d
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
def get_obj_pcs():
    with open('chairs.pkl', 'rb') as chairs_file:
        pcs = pickle.load(chairs_file)
    pcs = pcs[1:]
    b, n, _ = pcs.shape
    z = np.zeros((b, n, 3))
    ret = np.concatenate((pcs, z), axis=-1)
    return ret
def get_pc_from_pcbank():
    with open('cls_id2pcs.pkl', 'rb') as cls_id2pcsfile:
        cls_id2pcs = pickle.load(cls_id2pcsfile)
        cls_id2pcskeys=cls_id2pcs.keys()
        cls_idlist=[]
        pc=cls_id2pcs[524]
        for k in cls_id2pcskeys:
            cls_idlist.append(int(k))
            pc=torch.concat((pc,cls_id2pcs[k]),dim=0)
        cls_idlist=np.array(cls_idlist,dtype=np.int64)
        pc=pc[1:,...]
        pc=np.array(pc)
        return cls_idlist,pc



def sample_pc(pc):
    idx = np.random.choice(1024, 1024, replace=1024 < 1024)
    return pc[idx]

def show_pc(pt):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pt[..., :3])
    pc.colors = o3d.utility.Vector3dVector(pt[..., 3:])
    o3d.visualization.draw_geometries([pc])

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

def mean_color():
    return np.zeros((1, 3),dtype=np.float32)
def spin_pts(pts):
    mat = torch.tensor([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]],dtype=torch.float64)

    rotated_pc = torch.matmul(mat,pts[:,:3].T)
    rotated_pc=rotated_pc.T

    return torch.concat((rotated_pc,pts[:,3:]),dim=-1)
if __name__ == '__main__':

    class_to_idx_path = "class_to_idx_nr3d.txt"
    with open(class_to_idx_path, 'r') as class_to_idx_file:
        class_to_idx = json.load(class_to_idx_file)

    device = torch.device('cuda')
    # cls_idlist,pcs=get_pc_from_pcbank()
    # cls_idlist=torch.tensor(cls_idlist).to(device)
    # pcs=torch.tensor(pcs).to(device)
    # reg_pcs=pcs

    opcs = get_obj_pcs()
    pcs=[np.array(spin_pts(torch.tensor(op)) )for op in opcs]
    pcs=np.array(pcs)
    sampled_pcs = [sample_pc(pc) for pc in pcs]
    sampled_pcs=np.array(sampled_pcs)
    mean_rgb=mean_color()
    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
        unit_norm=True)
    reg_pcs = object_transformation(sampled_pcs)

    reg_pcs = torch.tensor(reg_pcs).float()
    reg_pcs=reg_pcs.to(device)

    args = parse_arguments()
    # Prepare GPU environment
    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"


    seed_training_code(args.random_seed)
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class    
    pad_idx = class_to_idx['pad']

    # Object-type classification
    class_name_list = []
    for cate in class_to_idx:
        class_name_list.append(cate)

    # tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    # class_name_tokens = tokenizer(class_name_list, return_tensors='pt', padding=True)
    # for name in class_name_tokens.data:
    #     class_name_tokens.data[name] = class_name_tokens.data[name].cuda()

    gpu_num = len(args.gpu.strip(',').split(','))
    model = ReferIt3DNet_transformer(args, n_classes, class_name_tokens=None, ignore_index=pad_idx)    
    
    model = model.to(device)
    # print(model)
    #load model param
    param_list=[
        {'params':model.language_encoder.parameters(),'lr':args.init_lr*0.1},
        {'params':model.refer_encoder.parameters(), 'lr':args.init_lr*0.1},
        {'params':model.object_encoder.parameters(), 'lr':args.init_lr},
        {'params':model.obj_feature_mapping.parameters(), 'lr': args.init_lr},
        {'params':model.box_feature_mapping.parameters(), 'lr': args.init_lr},
        {'params':model.language_clf.parameters(), 'lr': args.init_lr},
        {'params':model.object_language_clf.parameters(), 'lr': args.init_lr},
    ]
    if not args.label_lang_sup:
        param_list.append( {'params':model.obj_clf.parameters(), 'lr': args.init_lr})
    optimizer = optim.Adam(param_list,lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[40, 50, 60, 70, 80, 90], gamma=0.65)
    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not args.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            start_training_epoch = 0
            best_test_epoch = loaded_epoch
            best_test_acc = 0
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                best_test_acc))
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))


    net=model.object_encoder

    with torch.no_grad():
        net.eval()
        feat=net(reg_pcs)
    obj_feats = model.obj_feature_mapping(feat)
    cls_acc_mtr = AverageMeter()
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    class_name_tokens = tokenizer(class_name_list, return_tensors='pt', padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].cuda()
    label_lang_infos = model.language_encoder(**class_name_tokens)[0][:,0]

    B,N = feat.shape
    CLASS_LOGITS = torch.matmul(obj_feats, label_lang_infos.permute(1,0))
    # CLASS_LOGITS=CLASS_LOGITS.unsqueeze(0)
    class_labels=torch.zeros((B,),dtype=torch.int64)+87
    # class_labels=class_labels.unsqueeze(0)
    class_labels=class_labels.to(device)

    predictions = CLASS_LOGITS.argmax(dim=-1)
    n_obj=predictions.shape[0]
    similar_cls=[87]
    log10,top10=CLASS_LOGITS.topk(k=10)
    dif=[0]*n_obj
    for i in range(n_obj):
        if predictions[i]==87:
            for t10 in top10[i]:
                if t10 not in similar_cls:
                    dif[i]=(CLASS_LOGITS[i][87]-CLASS_LOGITS[i][t10])/CLASS_LOGITS[i][87]
                    break
    






            




    cls_b_acc, found_samples,n_obj_per_cls,n_err_per_cls,wrong_samples,wrong_valid_bn= cls_pred_stats(CLASS_LOGITS, class_labels, ignore_label=pad_idx)    
    
    cls_acc_mtr.update(cls_b_acc, B)
    print(cls_acc_mtr.avg)
