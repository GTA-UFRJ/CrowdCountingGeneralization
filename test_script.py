#%%
import torch
import sys
from mcnn_model import MCNN
from my_dataloader import CrowdDataset


paths_model = {
    'ucf_50':'/Users/lucascostafavaro/PycharmProjects/CrowdCounting/UCFCrowdCountingDataset_CVPR13/train_data/'
}


def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("mps")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    mae=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            # print(mae)
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))



if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    type_model = sys.argv[1]
    img_root = paths_model[type_model] + '/images'
    gt_dmap_root = paths_model[type_model] + "/ground_truth_npy"
    model_param_path='./checkpoints/'+type_model+'/epoch_'+sys.argv[2]+'.param'
    cal_mae(img_root,gt_dmap_root,model_param_path)
