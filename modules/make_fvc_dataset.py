from os import listdir
from os.path import join, isdir

import numpy as np
from matplotlib.image import imread
from scipy.misc import imresize

# reading fvc2002
FVC_PATH = 'dataset/FVC2002'
NEW_IMAGE_ROW = 181
NEW_IMAGE_COL = 181

db_path = join(FVC_PATH, 'Dbs')
folders = [f for f in listdir(db_path) if isdir(join(db_path, f))]
db_list = [np.zeros((110, 8, NEW_IMAGE_ROW, NEW_IMAGE_COL), dtype=np.uint8) for i in range(4)]

for folder in folders:
    print(folder)
    db_num = int(folder[2]) - 1

    for file in listdir(join(db_path, folder)):
        path = join(db_path, folder, file)
        finger_num, sample_num = [int(s) for s in file[:-4].split('_')]

        im = imread(path)
        im = imresize(im, (NEW_IMAGE_ROW, NEW_IMAGE_COL))
        db_list[db_num][finger_num - 1, sample_num - 1, :, :] = im

path = 'dataset/fvc_{}_{}.npz'.format(NEW_IMAGE_ROW, NEW_IMAGE_COL)
np.savez_compressed(path,
                    DB1=db_list[0],
                    DB2=db_list[1],
                    DB3=db_list[2],
                    DB4=db_list[3],
                    )
