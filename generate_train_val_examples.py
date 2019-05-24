import os
import numpy as np

data_path = "/ssd/zhangyiyang/DeepGlobe/land-train"
val_num = 100
train_file_path = "/ssd/zhangyiyang/DeepGlobe/land-train.txt"
val_file_path = "/ssd/zhangyiyang/DeepGlobe/land-val.txt"
all_file_path = "/ssd/zhangyiyang/DeepGlobe/land-all.txt"

file_list = os.listdir(data_path)
file_names = np.array([_ for _ in file_list if _.endswith('.png')])
np.random.shuffle(file_names)

train_list = file_names[:-val_num]
val_list = file_names[-val_num:]

with open(train_file_path, 'w') as f:
    for line in train_list:
        f.write(line + '\n')

with open(val_file_path, 'w') as f:
    for line in val_list:
        f.write(line + '\n')

with open(all_file_path, 'w') as f:
    for line in file_names:
        f.write(line + '\n')
