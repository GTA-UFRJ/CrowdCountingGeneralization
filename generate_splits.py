import pickle
import glob
import sys
import os
from sklearn.model_selection import train_test_split

path_model = {'ucf_50':'./data/UCF/data/',
'ucsd':'./data/UCSD/data/',
'mall':'./data/mall/data/',
'shang':'./data/ShanghaiTech/data/',
'drone':'./data/VisDrone2020-CC/data/'
}

def generate_splits(args = sys.argv):
    base_path = path_model[args[1]]
    n_splits = args[2]
    for n in range(int(n_splits)):
        x = os.listdir(f'{base_path}images')
        x_train, x_test = train_test_split(x,shuffle=True)
        with open(f'{base_path}train_splits/train_{n}.pkl','wb') as fp:
            pickle.dump(x_train, fp)
        with open(f'{base_path}test_splits/test_{n}.pkl','wb') as fp:
            pickle.dump(x_test, fp)


if __name__ == '__main__':
    generate_splits()
