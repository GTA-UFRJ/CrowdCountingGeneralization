import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM


#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(img,points,k=4):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    distances, locations = tree.query(points, k=k)

    print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = 0
            for j in range(k-1):
                sigma+=distances[i][j+1]
            sigma = 0.3*sigma/(k-1)
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density


if __name__=="__main__":
    root = '/data/mall/data'
    
    train = os.path.join(root,'images')
    path_sets = [train]
    
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth'))
        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        # this may change according to the dataset
        points = mat['annPoints']
        k = gaussian_filter_density(img,points)
        np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth_npy'), k)
    
