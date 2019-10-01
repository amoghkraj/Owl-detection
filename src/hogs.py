import cv2
import numpy as np
from cv2 import filter2D

class Descriptor:
    def __init__(self, imgs):
        self.winSize = (128,192)
        self.blockSize = (16,16)
        self.blockStride = (8,8)
        self.cellSize = (8,8)
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = 4.
        self.histogramNormType = 0
        self.L2HysThreshold = 2.0000000000000001e-01
        self.gammaCorrection = 0
        self.nlevels = 64
        self.hog = cv2.HOGDescriptor(self.winSize,self.blockSize,self.blockStride,self.cellSize,self.nbins,self.derivAperture,self.winSigma,self.histogramNormType,self.L2HysThreshold,self.gammaCorrection,self.nlevels)
        if len(imgs.shape) == 4:
            self.samples = np.array([self._getHOGs(img) for img in imgs])
        else:
            self.samples = self._getHOGs(imgs)


    def _getHOGs(self, img):
        return self.hog.compute(img).flatten()

