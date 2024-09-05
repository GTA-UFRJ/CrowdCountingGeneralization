import os
import torch
import torch.nn as nn
import sys
import time
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
import numpy as np
from glob import glob
import pickle

def collate_fn(batch):
    return tuple(zip(*batch))

paths_model = {
    #'ucf_50':'./data/UCF/data',
    #'shang':'./data/ShanghaiTech/data',
    #'ucsd':'./data/UCSD/data',
    'mall':'./data/mall/data',
    #'drone':'./data/VisDrone2020-CC/data'
}

def train(argv = sys.argv):
    type_model = argv[1]
    base_path = paths_model[argv[1]]
    n_split = argv[2]
    with open(f'{base_path}/train_splits/train_{n_split}.pkl',"rb") as fp:
        list_images = pickle.load(fp)

    torch.backends.cudnn.enabled=False
    device=torch.device("mps")

    mcnn=MCNN().to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-8,
                                momentum=0.95
    )
    img_root= f'{base_path}/images'
    gt_dmap_root=f"{base_path}/ground_truth_npy"
    dataset=CrowdDataset(img_root,list_images,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,)

    #training phase
    if not os.path.exists('./checkpoints/'+type_model):
        os.mkdir('./checkpoints/'+type_model)
    min_mae=10000
    min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(0,1):
        mcnn.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(dataloader):
        # for i, (img_b) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
#            print(gt_dmap.shape,img.shape)
            # forward propagation
            et_dmap=mcnn(img)
            # calculate loss
            loss=criterion(et_dmap,gt_dmap)
            if i % 100 == 0:
                print(loss)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))
        torch.save(mcnn.state_dict(),'./checkpoints/'+type_model+'/split_'+str(n_split)+".param")

if __name__=="__main__":
    train()