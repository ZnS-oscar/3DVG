import torch
import torch.nn.functional as F
import pickle
import numpy as np

def my_get_siamese_features(net, in_features, numbers):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    n_scenes,n_items = in_features.shape[:2]
    out_features = []
    for i in range(n_scenes):
        cc=net(in_features[i,:numbers[i]])
        dd=torch.ones(n_items,762).cuda()
        dd[:numbers[i]]=cc
        out_features.append(dd)
    out_features = torch.stack(out_features)
    return out_features

def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    # with torch.no_grad():
    #     net.eval()
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    # mypc=[np.zeros((1024,6)).tolist()]
    # mypct=torch.tensor(mypc).float().to(in_features.device)
    # myfeat=net(mypct)
    # mtmypct=mypct.repeat(48,1,1)
    # mtmyfeat=net(mtmypct)
    # myufeat=net(in_features[:,87])
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)

    # with open("cls_id2feat.pkl", 'rb') as cls_id2feat_file:
    #     loaded_data_dict = pickle.load(cls_id2feat_file)
    return out_features


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch
