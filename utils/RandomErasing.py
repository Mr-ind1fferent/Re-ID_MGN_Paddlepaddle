import random
import math
import cv2
import numpy as np


class RandomErasing:
    """Random erasing the an rectangle region in Image.
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    Args:
        sl: min erasing area region
        sh: max erasing area region
        r1: min aspect ratio range of earsing region
        p: probability of performing random erasing
    """

    def __init__(self,probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):

        self.p = probability
        self.s = (sl, sh)
        self.r = (r1, 1 / r1)

    def __call__(self, img):
        """
        perform random erasing
        Args:
            img: opencv numpy array in form of [w, h, c] range
                 from [0, 255]

        Returns:
            erased img
        """

        assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'

        if random.random() > self.p:
            return img

        else:
            while True:
                Se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
                re = random.uniform(*self.r)

                He = int(round(math.sqrt(Se * re)))
                We = int(round(math.sqrt(Se / re)))

                xe = random.randint(0, img.shape[1])
                ye = random.randint(0, img.shape[0])

                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    img[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img.shape[2]))

                    return img


