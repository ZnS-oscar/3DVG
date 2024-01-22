import torch
import pickle
import json

def read_cls_guess(n_obj,n_err,clsid):
    return n_obj[clsid],n_err[clsid],n_err[clsid]/n_obj[clsid]



if __name__ == '__main__':
    # original tot, using pc bank
    with open('tot_n_obj_per_clsori.pkl', 'rb') as tot_n_obj_per_clsorifile:
        n_obj_ori = pickle.load(tot_n_obj_per_clsorifile)
    with open('tot_n_err_per_clsori.pkl', 'rb') as tot_n_err_per_clsorifile:
        n_err_ori = pickle.load(tot_n_err_per_clsorifile)

    with open('tot_n_obj_per_cls.pkl', 'rb') as tot_n_obj_per_clsfile:
        n_obj = pickle.load(tot_n_obj_per_clsfile)
    with open('tot_n_err_per_cls.pkl', 'rb') as tot_n_err_per_clsfile:
        n_err = pickle.load(tot_n_err_per_clsfile)

    with open('class_to_idx.txt', 'r') as class_to_idxfile:
        class_to_idx = json.load(class_to_idxfile)
    

    # chair 87
    pc_bank_result=read_cls_guess(n_obj_ori,n_err_ori,87)
    new_obj_result=read_cls_guess(n_obj,n_err,87)
    
    print('hault here')


    