import os
import torch
import torch.nn as nn
import sys
import time
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
import numpy as np
from glob import glob

def collate_fn(batch):
    return tuple(zip(*batch))

paths_model = {
    'ucf_50':'/Users/lucascostafavaro/PycharmProjects/CrowdCounting/UCFCrowdCountingDataset_CVPR13/train_data/'
}

if __name__=="__main__":
    print('aqui')
    type_model = sys.argv[1]
    torch.backends.cudnn.enabled=False
    device=torch.device("mps")
    mcnn=MCNN().to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-8,
                                momentum=0.95)
    print('aqui')
    
    img_root= paths_model[type_model]+'/images'
    gt_dmap_root=paths_model[type_model]+"/ground_truth_npy"
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,)

    print('aqui')

    #training phase
    if not os.path.exists('./checkpoints/'+type_model):
        os.mkdir('./checkpoints/'+type_model)
    min_mae=10000
    min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    print('aqui')
    # x = glob('./checkpoints/*')
    # num_epoch = -1
    # if x != []:
    #     num_epoch = max([int(item[20:item.find('.param')]) for item in x])
    #     mcnn.load_state_dict(torch.load('./checkpoints/epoch_'+str(num_epoch)+'.param'))
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(0,20):
        mcnn.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(dataloader):
        # for i, (img_b) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
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
        torch.save(mcnn.state_dict(),'./checkpoints/'+type_model+'/epoch_'+str(epoch+1)+".param")

        mcnn.eval()
        mae=0
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            #print(et_dmap.data.sum(),'/',gt_dmap.data.sum())
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap
        if mae/len(dataloader)<min_mae:
            min_mae=mae/len(dataloader)
            min_epoch=epoch
        test_error_list.append(mae/len(dataloader))
        print("epoch:"+str(epoch)+" error:"+str(mae/len(dataloader))+" min_mae:"+str(min_mae)+" min_epoch:"+str(min_epoch))
        np.save('./checkpoints/'+type_model+'/epoch_'+str(epoch+1)+".npy",mae/len(dataloader))
        # vis.line(win=1,X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        # vis.line(win=2,X=epoch_list, Y=test_error_list, opts=dict(title='test_error'))
        # show an image
        # index=random.randint(0,len(test_dataloader)-1)
        # img,gt_dmap=test_dataset[index]
        # vis.image(win=3,img=img,opts=dict(title='img'))
        # vis.image(win=4,img=gt_dmap/(gt_dmap.max())*255,opts=dict(title='gt_dmap('+str(gt_dmap.sum())+')'))
        # img=img.unsqueeze(0).to(device)
        # gt_dmap=gt_dmap.unsqueeze(0)
        # et_dmap=mcnn(img)
        # et_dmap=et_dmap.squeeze(0).detach().cpu().numpy()
        # vis.image(win=5,img=et_dmap/(et_dmap.max())*255,opts=dict(title='et_dmap('+str(et_dmap.sum())+')'))
        #loss 97




        