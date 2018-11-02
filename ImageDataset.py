from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import Transform
from skimage import io

class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, rootdir = 'tiny_vid',dataType = 'train',dataAug = False):

        self.subdirs = [dir_name for dir_name in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, dir_name))]
        self.subdirs = sorted(self.subdirs)

        self.rootdir = rootdir
        self.dataType = dataType

        self.class_to_idx = {val:i for i,val in enumerate(self.subdirs)}
        self.idx_to_class = {self.class_to_idx[key]:key for key in self.class_to_idx.keys()}

        self.bbox_dic = self.get_bbox()

        if(not (dataType == 'train')):
            dataAug = False

        self.dataAug = dataAug


        self.transformList = [
        Transform.RandomCrop(100,doCrop=self.dataAug),
        Transform.Rescale(224),
        Transform.ToTensor()
        ]




    def __len__(self):
        if(self.dataType == 'train'):
            return 150 * 5
        else:
            return 30 * 5


    def __getitem__(self, idx):
        if(self.dataType == 'train'):
            class_idx = int(idx / 150)
            assert class_idx in (0,1,2,3,4)
            image_idx = idx % 150
        else:
            class_idx = int(idx / 30)
            assert class_idx in (0, 1, 2, 3, 4)
            image_idx = idx % 30 + 150

        imageName = str(image_idx+1).zfill(6) + '.JPEG'
        imagePath = os.path.join(self.rootdir,self.idx_to_class[class_idx],imageName)

        image = io.imread(imagePath)



        image_class = class_idx
        bbox = self.bbox_dic[class_idx][image_idx,:].reshape(4)

        sample = {'image': image, 'label': image_class,'bbox':bbox}
        sample = self.dataTransform(sample)
        # if self.transform:
        #     sample = self.transform(sample)

        return sample
    def get_bbox(self):
        bbox_dic = {}
        for val in self.subdirs:
            txtName = val + '_gt.txt'
            bbox_dic[self.class_to_idx[val]] = self.get_bbox_array(os.path.join(self.rootdir,txtName))
        return bbox_dic
    def get_bbox_array(self,filePath):
        with open(filePath) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = lines[i].strip('\n')
        resList = []
        for val in lines:
            val = val.split(' ')
            del val[0]
            resList.append(val)
        resNp = np.array(resList).astype(float)

        #change the data form to (x_min,y_min,w,h)
        resNp[:, 2] = resNp[:, 2] - resNp[:, 0]
        resNp[:, 3] = resNp[:, 3] - resNp[:, 1]

        return resNp/127.0 #normalize to 0.0 - 1.0

    def dataTransform(self,sample):
        for trans in self.transformList:
            sample = trans(sample)
        sample['image'] = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(sample['image'])
        return sample






