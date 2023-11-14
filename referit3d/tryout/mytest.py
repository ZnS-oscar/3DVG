import pandas as pd
# 这里直接使用gbk格式打开原始文件
df = pd.read_csv('/workspace/data_zoo/referit3dlang/nr3d_new.csv', encoding='gbk')
# 将数据重新保存为utf-8
df.to_csv('/workspace/data_zoo/referit3dlang/nr3d_new_u.csv', encoding='utf-8', index=False)
