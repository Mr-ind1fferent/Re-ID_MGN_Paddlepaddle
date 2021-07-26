import PIL
import paddle
# from paddle.vision import transforms
from paddle.vision.transforms import *
from paddle.io import Dataset, BatchSampler
from PIL import Image
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import os
import re
from paddle.vision import transforms
from paddle.io import DataLoader


def default_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ZD_dataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, transform, dtype, data_path):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(ZD_dataset, self).__init__()

        self.transform = transform
        # self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        path = self.imgs[index]
        target = self._id2label[self.id(path)]
        # target = paddle.to_tensor(data=target, dtype='float32', stop_gradient=False)
        img = Image.open(open(path, 'rb')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = paddle.to_tensor(data=img, dtype='float32', stop_gradient=False)

        return img, target

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        # param file_path :unix style file path
        # return : camera id
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):

        # return :camera id list corresponding to dataset image paths
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpep|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root + '/', f)  # windows下须加'/'，linux下则不需要
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])


class Data():
    def __init__(self):
        train_transform = transforms.Compose([
            Resize((384, 128), interpolation='bicubic'),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5)
        ])

        test_transform = transforms.Compose([
            Resize((384, 128), interpolation='bicubic'),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = ZD_dataset(train_transform, 'train', opt.data_path)
        self.testset = ZD_dataset(test_transform, 'test', opt.data_path)
        self.queryset = ZD_dataset(test_transform, 'query', opt.data_path)
        self.train_loader = paddle.io.DataLoader(dataset=self.trainset,
                                                 batch_sampler=BatchSampler(
                                                                            sampler=RandomSampler(self.trainset,
                                                                                                  batch_id=opt.batchid,
                                                                                                  batch_image=opt.batchimage),
                                                                            batch_size=16)
                                                 , num_workers=8)
        self.test_loader = paddle.io.DataLoader(dataset=self.testset, batch_size=opt.batchtest, num_workers=8)
        self.query_loader = paddle.io.DataLoader(dataset=self.queryset, batch_size=opt.batchtest, num_workers=8)
        if opt.mode =='vis':
            self.query_image = test_transform(default_loader(opt.query_image))