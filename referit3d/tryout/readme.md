v1.0
23-10-27
1. 增加tryout/find_obj_for_cls.py 用于为每一个cls寻找5个obj
2. 在train_referit3d.py增加   ` #gen class label txt by liu if not args.genclasslabeltxt:`, 用于生成class_label_dict.txt, 每一个场景的`list[obj_id]=cls_id`