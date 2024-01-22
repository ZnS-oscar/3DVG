import h5py
import torch
import pickle
# 打开H5文件
f = h5py.File('h5_files/split1_nobg/test_objectdataset.h5', 'r')

# 查看所有主键
print(f.keys())

# 读取数据
data = f['data'][:]
labels = f['label'][:]
unique,count=torch.unique(torch.tensor(labels),return_counts=True)
'''
unique
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
       dtype=torch.int32)

count
tensor([ 14+45+15+68+109+34+27+41+57+44+26+20+21+40+
         17])

Class
 0Bag 1Bed 2Bin 3Box 4Cabinet 5Chair 6Desk 7Display 8Door 9Pillow 10Shelf 11Sink 12Sofa 13Table 14Toilet
454~479
 pillow 480~499
 sink:500~520
sofa:521~560
toilet:561~eof

 '''

with open('table.pkl','wb') as file:
	pickle.dump(data[454:479,...], file)
# 关闭文件
f.close()