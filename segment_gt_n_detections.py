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
from boundingBoxDrawer import drawBoxes
from IPython import display
import count_detections

parser = argparse.ArgumentParser(description='divide bounding boxes into different \
                                 segments based on segment ground truth falls into')

parser.add_argument('--network', help='folder pattern for network to analyse',required=True)
parser.add_argument('--frameWidth', help='video frame width',required=True)
parser.add_argument('--frameHeight', help='video frame height',required=True)
parser.add_argument('--nrows', help='Number segments along height',required=True)
parser.add_argument('--ncols', help='Number segments along width',required=True)
parser.add_argument('--groundTruth', help='1 for groundtruth folders, 0 otherwise',required=True,default='0')

def processArguments(args):
    detfolders = glob(args.network)

    frameWidth = int(args.frameWidth)
    frameHeight = int(args.frameHeight)
    detSubfolder = '0_0-{}_{}'.format(frameWidth,frameHeight) # subfolder inside network detections folder

    nrows = int(args.nrows)
    ncols = int(args.ncols)
    groundTruth = int(args.groundTruth)
    segmentsDF = generateSegments(nrows,ncols,frameHeight,frameWidth)

    for i,df in enumerate(detfolders):#go through list of detection folders that match pattern
        #if i == 72:
        segment_gt_n_detections(df,detSubfolder,frameWidth,frameHeight,segmentsDF,groundTruth=groundTruth)
            
            #save number of detections for each segment in csv file
        count_detections.save_counts(os.sep.join([df,detSubfolder,'{}r{}c'.format(nrows,ncols)]))
        print('{}/{}: {}'.format(i+1,len(detfolders),ntpath.basename(df)))
        # print('{}/{}: {}, frames = {}'.format(
        #     i, len(detfolders),df, len(glob(df + '/' + detSubfolder + '/*.txt'))
        # ))
def generateSegments(nrows,ncols,frameHeight,frameWidth):
    yPoints = np.linspace(start=0,stop=frameHeight,num=nrows+1,dtype=np.int,endpoint=True)
    xPoints = np.linspace(start=0,stop=frameWidth,num=ncols+1,dtype=np.int,endpoint=True)
    #create segments from img
    leftTop_rightBottom = []
    for y in range(len(yPoints) - 1):
        for x in range(len(xPoints) - 1):
            segmentFolder = '{}r{}c/{}_{}-{}_{}'.format(nrows,ncols,xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1])
            
            leftTop_rightBottom.append([xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1],
                                        segmentFolder])
    segmentsDF = pd.DataFrame(leftTop_rightBottom,columns=['left','top','right','bottom','foldername'])
    return segmentsDF
def remakeSegmentDirectoryPath(path,segmentFoldername):
    if os.path.exists(path + '/' + segmentFoldername):
        shutil.rmtree(path + '/' + segmentFoldername)
    os.makedirs(os.sep.join([path,segmentFoldername,'analysis']), exist_ok=True)
# def generateSegments(nrows,ncols,frameHeight,frameWidth,pfolder,detfolders):
#     yPoints = np.linspace(start=0,stop=frameHeight,num=nrows+1,dtype=np.int,endpoint=True)
#     xPoints = np.linspace(start=0,stop=frameWidth,num=ncols+1,dtype=np.int,endpoint=True)
#     #create segments from img
#     leftTop_rightBottom = []
#     for y in range(len(yPoints) - 1):
#         for x in range(len(xPoints) - 1):
#             segmentFolder = '{}r{}c/{}_{}-{}_{}'.format(nrows,ncols,xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1])
            
#             for detfolder in detfolders:
#                 detfolder = pfolder + detfolder
#                 if os.path.exists(detfolder + '/' + segmentFolder):
#                     shutil.rmtree(detfolder + '/' + segmentFolder)
                
#                 os.makedirs(os.sep.join([detfolder,segmentFolder,'analysis']), exist_ok=True)

#             leftTop_rightBottom.append([xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1],
#                                         segmentFolder])
#     segmentsDF = pd.DataFrame(leftTop_rightBottom,columns=['left','top','right','bottom','foldername'])
#     return segmentsDF

def read_detections(fname,cols):
    '''
    Reads the csv of detections/ground truth files and computes the centers of 
    the bounding boxes
    '''
    df = pd.read_csv(fname,sep=' ',names=cols,engine='python')
    df['xcentre'] = (df.right + df.left) / 2.0
    df['ycentre'] = (df.bottom + df.top) / 2.0
    return df
def writeSegmentData(folderName,segmentData,frameName,detections,detectionsList,groundTruth=False):
    writtenDetections = []
#    print(folderName,segmentData.foldername,frameName)
    
    with open(folderName + '/' + segmentData.foldername + '/' + frameName, 'a+') as detFile:
        for _,detData in detections.iterrows():
            if segmentData.left <= detData.xcentre and segmentData.top <= detData.ycentre and \
                segmentData.right > detData.xcentre and segmentData.bottom > detData.ycentre:
                if groundTruth:
                    detStr = '{} {} {} {} {}\n'.format(detData['class'],detData.left,detData.top,detData.right,detData.bottom)    
                else:
                    detStr = '{} {} {} {} {} {}\n'.format(detData['class'],detData.conf,detData.left,detData.top,detData.right,detData.bottom)
                if detStr not in detectionsList:#check if current detections have not been written in a file for this frame
                    detFile.write(detStr)
                    writtenDetections.append(detStr)
    return writtenDetections
    
def segment_gt_n_detections(parentFolder,detectionFolder,frameWidth,frameHeight,segmentsDF,groundTruth=False):
    detFiles = glob(os.sep.join([parentFolder, detectionFolder, '/*.txt']))
    detTXTFolder = os.sep.join([parentFolder,detectionFolder])
    for segmentFolder in segmentsDF['foldername']:
        remakeSegmentDirectoryPath(detTXTFolder,segmentFolder)
    
    for fidx,detFile in enumerate(detFiles):
        frameName = ntpath.basename(detFile)
        if groundTruth:
            detDF = read_detections(detFile, ['class','left','top','right','bottom'])
        else:
            detDF = read_detections(detFile,['class','conf','left','top','right','bottom'])
        
        detList = [] # list of frames already written into a particular segment
        
        for _,segmentData in segmentsDF.iterrows():
            detList = detList + writeSegmentData(detTXTFolder,segmentData,frameName,detDF,detList,groundTruth=groundTruth)

        # print(fidx,frameName[:-4])

if __name__ == '__main__':
    args = parser.parse_args()
    processArguments(args)
    # segment_gt_n_detections(pfolder,gtfolder,detfolders,frameWidth,frameHeight,nrows,ncols,dthreshold)