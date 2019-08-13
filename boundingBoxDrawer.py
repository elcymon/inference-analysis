# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rc('image',cmap='plasma')

from glob import glob
#import numpy as np
import matplotlib.pyplot as plt

#import pathlib
import os
import ntpath
#from segment_gt_n_detections import read_detections
import re

def read_detections(fname,cols):
    '''
    Reads the csv of detections/ground truth files and computes the centers of 
    the bounding boxes
    '''
    df = pd.read_csv(fname,sep=' ',names=cols,engine='python')
    df['xcentre'] = (df.right + df.left) / 2.0
    df['ycentre'] = (df.bottom + df.top) / 2.0
    return df

def extractSegment(segmentStr):
    seg = re.findall(r'\d+',segmentStr)
    return [int(s) for s in seg]
    
def drawBoxes(gtFile,detFile,imgSize):
    '''
    In:
        gtFile = groundtruth file
        detFile = detection file
        imgSize = (xmin,ymin,xmax,ymax)
    '''
    gtDF = read_detections(gtFile,['class','left','top','right','bottom'])
    detDF = read_detections(detFile,['class','conf','left','top','right','bottom'])
    
    
    fig,ax = plt.subplots()
#    print("gtfile: " + gtFile)
#    print(os.sep.split(gtFile))
    gtSegment = extractSegment(gtFile.split(os.sep)[-2])
    detSegment = extractSegment(detFile.split(os.sep)[-2])
    print('image size',imgSize)
    print('ground truths: ',gtDF.shape[0], 'in segment',gtSegment)
#    print(gtDF)
    print('detection: ',detDF.shape[0],'in segement ',detSegment)
#    print(detDF)
    frameNo = detFile.split(os.sep)[-1]
    frameNo = frameNo[:-4]
    
    xmin,ymin,xmax,ymax = imgSize
    plt.xlim(xmin,xmax)
    plt.ylim(ymax,ymin)
#    plt.axis([-1000,1000,-1000,1000])
    if gtSegment == detSegment:
        l,t,r,b = gtSegment
        rect = mpl.patches.Rectangle((l,t),r-l,b-t,linestyle='--',linewidth=1,
                                     edgecolor='k',facecolor='none',label='segment')
        ax.add_patch(rect)
    else:
        l,t,r,b = gtSegment
        rect = mpl.patches.Rectangle((l,t),r-l,b-t,linestyle='--',linewidth=1,
                                     edgecolor='b',facecolor='none',label='gtsegment')
        ax.add_patch(rect)
        
        l,t,r,b = detSegment
        
        rect = mpl.patches.Rectangle((l,t),r-l,b-t,linestyle='--',linewidth=1,
                                     edgecolor='g',facecolor='none',label='detSegment')
        ax.add_patch(rect)
    
#    print(gtSegment)
#    print(detSegment)
#    print(imgSize)
    for i,r in gtDF.iterrows():
        if i == 0:
            rect = mpl.patches.Rectangle((r.left,r.top),\
                                         r.right-r.left,r.bottom-r.top,\
                                         linewidth=2,edgecolor='b',\
                                         facecolor='none',label='GT')
        else:
            rect = mpl.patches.Rectangle((r.left,r.top),r.right - r.left,\
                                         r.bottom - r.top,linewidth=2,\
                                         facecolor='none',edgecolor='b')
        ax.add_patch(rect)
    
    for i,r in detDF.iterrows():
        if i == 0:
            rect = mpl.patches.Rectangle((r.left,r.top),\
                                         r.right - r.left,r.bottom - r.top,\
                                         linewidth=2,edgecolor='g',\
                                         facecolor='none',label='Detection')
        else:
            rect = mpl.patches.Rectangle((r.left,r.top),r.right-r.left,\
                                         r.bottom-r.top,linewidth=2,\
                                         facecolor='none',edgecolor='g')
        ax.add_patch(rect)
        
    ax.set_title(frameNo)
    plt.legend()
    plt.show()
    
    
