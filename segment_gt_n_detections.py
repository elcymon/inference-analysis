# -*- coding: utf-8 -*-
# this function loops through all Ground Truth frames and Detections
# and distributes both into appropriate segments based on segment ground truth
# falls into. The decision is made based on distance between detection box
# centre and nearest centre of ground truth bounding boxes

import argparse
import os
from glob import glob
import ntpath
import pandas as pd
import numpy as np
import shutil


parser = argparse.ArgumentParser(description='divide bounding boxes into different \
                                 segments based on segment ground truth falls into')

parser.add_argument('--gtfolder', help='Path to ground truths text files')
parser.add_argument('--detfolder', help='Path to detections text files')
parser.add_argument('--frameWidth', help='video frame width')
parser.add_argument('--frameHeight', help='video frame height')
parser.add_argument('--nrows', help='Number segments along height')
parser.add_argument('--ncols', help='Number segments along width')
parser.add_argument('--dthreshold',help='Threshold for pixel distance between centres')

args = parser.parse_args()
pfolder = '/home/elcymon/litter-detection/darknet/videos/'
gtfolder = pfolder + '20190111GOPR9027half-yolov3-litter_10000-th0p1-nms0p0-iSz608-nosegment'
detfolder = pfolder + '20190111GOPR9027half-mobilenetSSD-10000-th0p7-nms0p0-iSz216-nosegment'
frameWidth = 960
frameHeight = 540
nrows = 3
ncols = 4
dthreshold = 20

def generateSegments(nrows,ncols,frameHeight,frameWidth,gtfolder,detfolder):
    yPoints = np.linspace(start=0,stop=frameHeight,num=nrows,dtype=np.int,endpoint=True)
    xPoints = np.linspace(start=0,stop=frameWidth,num=ncols,dtype=np.int,endpoint=True)
    #create segments from img
    leftTop_rightBottom = []
    for y in range(len(yPoints) - 1):
        for x in range(len(xPoints) - 1):
            segmentFolder = '{}_{}-{}_{}'.format(xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1])
            
            if os.path.exists(gtfolder + '/' + segmentFolder):
                shutil.rmtree(gtfolder + '/' + segmentFolder)
            
            os.mkdir(gtfolder + '/' + segmentFolder)
            os.mkdir(gtfolder + '/' + segmentFolder + '/analysis')
            
            if os.path.exists(detfolder + '/' + segmentFolder):
                shutil.rmtree(detfolder + '/' + segmentFolder)
                
            os.mkdir(detfolder + '/' + segmentFolder)
            os.mkdir(detfolder + '/' + segmentFolder + '/analysis')
            
            leftTop_rightBottom.append([xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1],
                                        segmentFolder])
    segmentsDF = pd.DataFrame(leftTop_rightBottom,columns=['left','top','right','bottom','foldername'])
    return segmentsDF

def segment_gt_n_detections(pfolder,gtfolder,detfolder,frameWidth,frameHeight,nrows,ncols,dthreshold):
    gtfiles = glob(gtfolder + '/*.txt')
    segmentsDF = generateSegments(nrows,ncols,frameHeight,frameWidth,gtfolder,detfolder)
    
    for fidx,gtfile in enumerate(gtfiles):
        frameName = ntpath.basename(gtfile)
        print(fidx,frameName)
        detfile = detfolder + '/' + frameName
        detDF = pd.read_csv(detfile, sep=' ', names=['class','conf','left','top','right','bottom'], engine='python')
        detDF['xcentre'] = (detDF.right + detDF.left) / 2.0
        detDF['ycentre'] = (detDF.bottom + detDF.top) / 2.0
        
        gtDF = pd.read_csv(gtfile, sep=' ', names=['class','left','top','right','bottom'], engine='python')
        gtDF['xcentre'] = (gtDF.right + gtDF.left) / 2.0
        gtDF['ycentre'] = (gtDF.bottom + gtDF.top) / 2.0
        for j,segmentData in segmentsDF.iterrows():
            with open(gtfolder + '/' + segmentData.foldername + '/' + frameName,'a+') as gt:
                for i,gtData in gtDF.iterrows():
                    if segmentData.left <= gtData.xcentre and segmentData.top <= gtData.ycentre and \
                        segmentData.right > gtData.xcentre and segmentData.bottom > gtData.ycentre:
                            gt.write('{} {} {} {} {}\n'.format(gtData['class'],gtData.left,gtData.top,gtData.right,gtData.bottom))
#            segment = segmentsDF.loc[(segmentsDF['left'] <= rowData['xcentre']) & \
#                                    (segmentsDF['top'] <= rowData['ycentre']) & \
#                                    (segmentsDF['right'] > rowData['xcentre']) & \
#                                    (segmentsDF['bottom'] > rowData['ycentre']),:]
            
        
#        return  gtDF,detDF,gtfiles
#    return gtfiles
#segmentsDF = generateSegments(nrows,ncols,frameHeight,frameWidth,gtfolder,detfolder)
#gtDF,detDF,gtfiles= \
segment_gt_n_detections(pfolder,gtfolder,detfolder,frameWidth,frameHeight,nrows,ncols,dthreshold)