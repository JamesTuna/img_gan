from scipy.misc import *

import sys
import numpy as np
import os
from progress.bar import Bar

th=0.1
def imageUpScale(inFolder, inImg, outFolder, factor=8):
    inImgPath = os.path.join(inFolder, inImg)
    img = imread(inImgPath, mode='L')/255.0
    img[img>th]=1
    img[img<th]=0
    h=len(img)
    w=len(img[0])
    newImg=np.zeros((2048,2048))
    for i in xrange(h):
        for j in xrange(w):
            newImg[i*8:(i+1)*8, j*8:(j+1)*8]=img[i,j]
    outImgPath = os.path.join(outFolder,'upscaled_'+inImg)
    imsave(outImgPath, newImg)

inFolder = sys.argv[1]
outFolder= sys.argv[2]

for dirname, dirnames, filenames in os.walk(inFolder):
    bar=Bar("Upscaling Image", max=len(filenames))
    for f in xrange(0, len(filenames)):
        imageUpScale(inFolder, filenames[f], outFolder)
        bar.next()
bar.finish()