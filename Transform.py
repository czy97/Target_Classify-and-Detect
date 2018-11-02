from skimage import transform
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image= sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w),mode='constant')

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        sample['image'] = img

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size = 100,thres = 0.7,doCrop = True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.thres = thres
        self.doCrop = doCrop


    def __call__(self, sample):
        if(not self.doCrop):
            return sample

        cropRatio = 4
        original_or_crop = np.random.randint(0, cropRatio)
        if(original_or_crop == 0):
            return sample


        image, label, bbox = sample['image'], sample['label'], sample['bbox']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        y_leftTop = np.random.randint(0, h - new_h)
        x_leftTop = np.random.randint(0, w - new_w)

        #try five times to crop,if failure,not crop
        for i in range(5):

            ratio,newbbox = self.calTargetProportion(bbox,(x_leftTop,y_leftTop))
            if(ratio > self.thres):
                image = image[y_leftTop: y_leftTop + new_h,
                        x_leftTop: x_leftTop + new_w]
                sample['image'] = image
                sample['label'] = label
                sample['bbox'] = newbbox
                return sample

        return sample
    def calTargetProportion(self,bbox,new_leftTop):
        x_min_bbox = int(bbox[0] * 127)
        y_min_bbox = int(bbox[1] * 127)
        x_max_bbox = int((bbox[0] + bbox[2]) * 127)
        y_max_bbox = int((bbox[1] + bbox[3]) * 127)

        x_min_crop = new_leftTop[0]
        y_min_crop = new_leftTop[1]
        x_max_crop = new_leftTop[0] + self.output_size[1]
        y_max_crop = new_leftTop[1] + self.output_size[0]

        W = min(x_max_bbox, x_max_crop) - max(x_min_bbox, x_min_crop)
        H = min(y_max_bbox, y_max_crop) - max(y_min_bbox, y_min_crop)

        if W <= 0 or H <= 0:
            return (0.0,None)

        cross = W * H
        targetArea = ((x_max_bbox - x_min_bbox) * (y_max_bbox - y_min_bbox))

        #calculate the new location of the bbox according to crop area
        x_max_bbox = min(x_max_crop,x_max_bbox) - x_min_crop
        y_max_bbox = min(y_max_crop,y_max_bbox) - y_min_crop
        x_min_bbox = max(0,x_min_bbox - x_min_crop)
        y_min_bbox = max(0,y_min_bbox - y_min_crop)

        x_min = 1.0 * x_min_bbox / (self.output_size[1] - 1)
        y_min = 1.0 * y_min_bbox / (self.output_size[0] - 1)
        w = 1.0 * (x_max_bbox - x_min_bbox) / (self.output_size[1] - 1)
        h = 1.0 * (y_max_bbox - y_min_bbox) / (self.output_size[0] - 1)
        bbox_array = np.array([x_min,y_min,w,h]).reshape(4)

        return ((1.0 * cross / targetArea),bbox_array)




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label , bbox = sample['image'], sample['label'],sample['bbox']
        image = image.astype(float)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).type(torch.float32),
                'label': torch.from_numpy(np.array(label)).type(torch.long),
                'bbox': torch.from_numpy(bbox).type(torch.float32)
                }