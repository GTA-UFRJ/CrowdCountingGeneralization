import torch
import sys
import pickle
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
import numpy as np


paths_model = {
    'ucf_50':'./data/UCF/data/',
    'shang':'./data/ShanghaiTech/data/',
    'ucsd':'./data/UCSD/data/',
    'mall': './data/mall/data/',
    'drone':'./data/VisDrone2020-CC/data/'
}


def cal_mae(img_root,list_image,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("mps")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,list_image,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    mae=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            #print(et_dmap.data.sum())
            #print(gt_dmap.data.sum())
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            # print(mae)
            del img,gt_dmap,et_dmap
    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))
    return mae/len(dataloader)



def test(args = sys.argv):
    torch.backends.cudnn.enabled=False
    type_model = args[1]
    type_eval = args[2]
    n_split = args[3]
    base_path = paths_model[type_eval]
    with open(f'{base_path}test_splits/test_{n_split}.pkl','rb') as fp:
        img_list = pickle.load(fp)

    img_root = f'{base_path}images'
    gt_dmap_root = f"{base_path}ground_truth_npy"
    model_param_path='./checkpoints/'+type_model+'/split_'+n_split+'.param'
    print(model_param_path)
    return cal_mae(img_root,img_list,gt_dmap_root,model_param_path)
#    np.save('./checkpoints/'+type_model+'_'+type_eval+'.npy',np.array(result))

if __name__ == '__main__':
    test()
