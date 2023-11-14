import pandas as pd

# 读取CSV文件
df = pd.read_csv('referit3dlang/nr3d_1.csv')

# 将需要移动的行保存到变量中
row_to_move = df.loc[0]
# new_df = df.iloc[[2549], :]
# new_df.to_csv('referit3dlang/nr3d_1.csv', index=False)
# print(row_to_move)
# 删除需要移动的行
# df = df.drop(2)
# 将需要移动的行插入到第一行
df = pd.concat([row_to_move.to_frame().T, df], ignore_index=True)
df = pd.concat([row_to_move.to_frame().T, df], ignore_index=True)
df = pd.concat([row_to_move.to_frame().T, df], ignore_index=True)
df = pd.concat([row_to_move.to_frame().T, df], ignore_index=True)
df = pd.concat([row_to_move.to_frame().T, df], ignore_index=True)
# 将修改后的数据保存到CSV文件中
df.to_csv('referit3dlang/nr3d_1.csv', index=False)