import sys
import json

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data

if __name__ == '__main__':
    input_file_path = "class_label_dict.txt"
    with open(input_file_path, 'r') as class_label_dict_file:
        class_label_dict = json.load(class_label_dict_file)



    # scene:{cls_id:[obj_id]} scn_cls_id2obj_id.txt
    # 创建一个空的新字典
    new_dict = {}

    # 遍历原始字典
    for scan_id, class_id_list in class_label_dict.items():
        # 创建一个空的字典来存储 class_id 和相应的 obj_id 列表
        class_id_obj_id_dict = {}
        
        for obj_id, class_id in enumerate(class_id_list):
            # 检查 class_id 是否已经在字典中，如果不在，创建一个新的列表
            if class_id not in class_id_obj_id_dict:
                class_id_obj_id_dict[class_id] = []
            
            # 将 obj_id 添加到对应的 class_id 的列表中
            class_id_obj_id_dict[class_id].append(obj_id)
        
        # 将 class_id_obj_id_dict 添加到新字典中
        new_dict[scan_id] = class_id_obj_id_dict

    output_file_path = "scn_cls_id2obj_id.txt"
    with open(output_file_path, 'w') as scn_cls_id2obj_id_file:
        json.dump(new_dict, scn_cls_id2obj_id_file)


    scn_cls_id2obj_id_path = "scn_cls_id2obj_id.txt"
    with open(scn_cls_id2obj_id_path, 'r') as scn_cls_id2obj_id_file:
        scn_cls_id2obj_id = json.load(scn_cls_id2obj_id_file)

    # read cls2idx
    input_file_path2="class_to_idx.txt"
    with open(input_file_path2, 'r') as class_to_idx_file:
        class_to_idx = json.load(class_to_idx_file)

    # get 5 obj_id for each cls_id
    for_cls_id_get_obj_id={}
    
    cls_id_flag=[False]*900
    scene_ids=scn_cls_id2obj_id.keys()
    for scene_id in scene_ids:
        cls_id2obj_id=scn_cls_id2obj_id[scene_id]
        cls_ids=cls_id2obj_id.keys()
        # if scene_id=="scene0517_02":#cls_id=167
        #     print("attention!")
        for cls_id in cls_ids:
            obj_ids=cls_id2obj_id[cls_id]
            if cls_id_flag[int(cls_id)]:
                continue
            if cls_id not in for_cls_id_get_obj_id:
                for_cls_id_get_obj_id[cls_id]=[(scene_id,o_id) for o_id in obj_ids]
            else:
                for_cls_id_get_obj_id[cls_id]+=[(scene_id,o_id) for o_id in obj_ids]
<<<<<<< HEAD
            if len(for_cls_id_get_obj_id[cls_id])>=5:
                for_cls_id_get_obj_id[cls_id]=for_cls_id_get_obj_id[cls_id][:5]
=======
            if len(for_cls_id_get_obj_id[cls_id])>=50:
                for_cls_id_get_obj_id[cls_id]=for_cls_id_get_obj_id[cls_id][:50]
>>>>>>> feat_bank
                cls_id_flag[int(cls_id)]=True
     
    for_cls_id_get_obj_id_keys=for_cls_id_get_obj_id.keys()
    for cls_id1 in class_to_idx.values():
        if str(cls_id1) not in for_cls_id_get_obj_id_keys:
            print(cls_id1)
    # for cls_id2 in for_cls_id_get_obj_id_keys:
    #     obj_num=len(for_cls_id_get_obj_id[cls_id2])
    #     while obj_num<5:
    #         for_cls_id_get_obj_id[cls_id2].append(for_cls_id_get_obj_id[cls_id2][-1])
    #         obj_num+=1

        
    
    
    output_file_path = "for_cls_id_get_obj_id.txt"
    with open(output_file_path, 'w') as for_cls_id_get_obj_id_file:
        json.dump(for_cls_id_get_obj_id, for_cls_id_get_obj_id_file)






    
    