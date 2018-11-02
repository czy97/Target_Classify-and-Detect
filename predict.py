from Model import *
import torch
import os
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from functions import *
import numpy as np
import glob


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


paramPath = 'storedModels/SmoothL1Loss-sgd-bestParam.pth'
model = VggBn_Net()
model = model.to(device)
model.load_state_dict(torch.load(paramPath,map_location=device))
model.eval()


image_datasets = {x: ImageDataset(dataType = x) for x in ['train', 'test']}
train_len = len(image_datasets['train'])
test_len = len(image_datasets['test'])

dataloaders_dict = {}
dataloaders_dict['train'] = DataLoader(image_datasets['train'],batch_size=8, shuffle=False, num_workers=4)
dataloaders_dict['test'] = DataLoader(image_datasets['test'],batch_size=8, shuffle=False, num_workers=4)


phase = ['train','test']

'''
Get pred data of train and test data
'''
Res_dic = {'train':{},'test':{}}
Res_dic['train']['pred'] = np.zeros(750).astype(int)
Res_dic['train']['bboxPred'] = np.zeros((750,4)).astype(int)
Res_dic['test']['pred'] = np.zeros(150).astype(int)
Res_dic['test']['bboxPred'] = np.zeros((150,4)).astype(int)

with torch.set_grad_enabled(False):
    for phase_name in phase:
        count = 0
        for val in dataloaders_dict[phase_name]:

            inputs = val['image'].to(device)
            labels = val['label'].to(device)
            bbox = val['bbox'].to(device)

            N = bbox.size()[0]

            output_label, output_bbox = model(inputs)
            _, preds = torch.max(output_label, 1)


            if(torch.cuda.is_available()):
                preds = preds.cpu()
                output_bbox = output_bbox.cpu()
            pred_numpy = preds.numpy().reshape(N).astype(int)

            bboxPred_numpy = output_bbox.numpy()
            bboxPred_numpy = bboxPred_numpy * 127
            bboxPred_numpy[:, 2] = bboxPred_numpy[:, 0] + bboxPred_numpy[:, 2]
            bboxPred_numpy[:, 3] = bboxPred_numpy[:, 1] + bboxPred_numpy[:, 3]
            bboxPred_numpy.astype(int).reshape(N, 4)
            bboxPred_numpy = np.clip(bboxPred_numpy, 0, 127)

            Res_dic[phase_name]['pred'][count:count+N] = pred_numpy
            Res_dic[phase_name]['bboxPred'][count:count + N,:] = bboxPred_numpy

            count += N


Res_dic['train']['pred'] = Res_dic['train']['pred'].reshape(750,1)
Res_dic['test']['pred'] = Res_dic['test']['pred'].reshape(150,1)


'''
Get each class's predict data 
And store these data into specific file,such as 'bird_predict.txt
'''
prefix = 'predictFile'
rootdir = 'tiny_vid'
class_names = [dir_name for dir_name in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, dir_name))]
class_names = sorted(class_names)
# class_names = ['car', 'turtle', 'dog', 'lizard', 'bird']
dataNum = {'train':150,'test':30}
res = {}
for i,class_name in enumerate(class_names):
    tmp_array_dic = {'train':{},'test':{}}
    for phase_name in phase:
        temNum = dataNum[phase_name]
        tmp_array_dic[phase_name] = np.concatenate((Res_dic[phase_name]['pred'][i*temNum:(i+1)*temNum]
                                                    ,Res_dic[phase_name]['bboxPred'][i*temNum:(i+1)*temNum,:]),1)
    tmp_array = np.concatenate((tmp_array_dic['train'],tmp_array_dic['test']),0)
    res[class_name] = tmp_array

    txtFile = os.path.join(prefix, class_name + '_predict.txt')
    res[class_name] = res[class_name].astype(str)
    f1 = open(txtFile, 'w')
    list_tmp = res[class_name].tolist()
    for line in list_tmp:
        f1.write(' '.join(line))
        f1.write('\n')
    f1.close()


'''
Plot the predicted bbox
'''
import cv2
def getBBox_data(filePath):
    with open(filePath) as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            lines[i] = lines[i].strip('\n').split(' ')

    res = np.array(lines).astype(int)

    return res

gtfile_prefix = 'tiny_vid'
for i,class_name in enumerate(class_names):
    store_dir = os.path.join(prefix,class_name+'_predict')
    if(not os.path.exists(store_dir)):
        os.makedirs(store_dir)


    image_gt_file = os.path.join(gtfile_prefix,class_name+'_gt.txt')
    image_predict_bbox_file = os.path.join(prefix,class_name+'_predict.txt')

    image_gt = getBBox_data(image_gt_file)
    image_predict_bbox = getBBox_data(image_predict_bbox_file)

    images_dir = os.path.join(gtfile_prefix, class_name)
    imagePath_list = glob.glob(os.path.join(images_dir,'*.JPEG'))

    for i,imagepath in enumerate(imagePath_list):
        gt_val = ((image_gt[i][1],image_gt[i][2]),(image_gt[i][3],image_gt[i][4]))
        predict_bbox_val = ((image_predict_bbox[i][1],image_predict_bbox[i][2]),(image_predict_bbox[i][3],image_predict_bbox[i][4]))

        img = cv2.imread(imagepath, 1)
        cv2.rectangle(img, gt_val[0], gt_val[1], (0, 255, 0), 3)
        cv2.rectangle(img, predict_bbox_val[0], predict_bbox_val[1], (255, 0, 0), 3)

        storeImagePath = os.path.join(store_dir,os.path.basename(imagepath))
        cv2.imwrite(storeImagePath,img)



















