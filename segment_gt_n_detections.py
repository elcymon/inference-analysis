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
detfolder = pfolder + '20190111GOPR9027half-yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128-nosegment/0_0-960_540'
frameWidth = 960
frameHeight = 540
nrows = 3
ncols = 4
dthreshold = 10

def generateSegments(nrows,ncols,frameHeight,frameWidth,gtfolder,detfolder):
    yPoints = np.linspace(start=0,stop=frameHeight,num=nrows+1,dtype=np.int,endpoint=True)
    xPoints = np.linspace(start=0,stop=frameWidth,num=ncols+1,dtype=np.int,endpoint=True)
    #create segments from img
    leftTop_rightBottom = []
    for y in range(len(yPoints) - 1):
        for x in range(len(xPoints) - 1):
            segmentFolder = '{}r{}c/{}_{}-{}_{}'.format(nrows,ncols,xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1])
            
            if os.path.exists(gtfolder + '/' + segmentFolder):
                shutil.rmtree(gtfolder + '/' + segmentFolder)
            os.makedirs(os.sep.join([gtfolder,segmentFolder,'analysis']), exist_ok=True)
#            os.mkdir(gtfolder + '/' + segmentFolder)
#            os.mkdir(gtfolder + '/' + segmentFolder + '/analysis')
            
            if os.path.exists(detfolder + '/' + segmentFolder):
                shutil.rmtree(detfolder + '/' + segmentFolder)
                
            os.makedirs(os.sep.join([detfolder,segmentFolder,'analysis']), exist_ok=True)
#            os.mkdir(detfolder + '/' + segmentFolder)
#            os.mkdir(detfolder + '/' + segmentFolder + '/analysis')
            
            leftTop_rightBottom.append([xPoints[x], yPoints[y], xPoints[x+1], yPoints[y+1],
                                        segmentFolder])
    segmentsDF = pd.DataFrame(leftTop_rightBottom,columns=['left','top','right','bottom','foldername'])
    return segmentsDF

def read_detections(fname,cols):
    '''
    Reads the csv of detections/ground truth files and computes the centers of 
    the bounding boxes
    '''
    df = pd.read_csv(fname,sep=' ',names=cols,engine='python')
    df['xcentre'] = (df.right + df.left) / 2.0
    df['ycentre'] = (df.bottom + df.top) / 2.0
    return df

def segment_gt_n_detections(pfolder,gtfolder,detfolder,frameWidth,frameHeight,nrows,ncols,dthreshold):
    gtfiles = glob(gtfolder + '/*.txt')
    segmentsDF = generateSegments(nrows,ncols,frameHeight,frameWidth,gtfolder,detfolder)
    
    for fidx,gtfile in enumerate(gtfiles):
        frameName = ntpath.basename(gtfile)
        detfile = detfolder + '/' + frameName
        detDF = read_detections(detfile,['class','conf','left','top','right','bottom'])
        
        gtDF = read_detections(gtfile, ['class','left','top','right','bottom'])
        maxDetcxy = 'nan'
        for j,segmentData in segmentsDF.iterrows():
            FP = False
            with open(gtfolder + '/' + segmentData.foldername + '/' + frameName,'a+') as gt:
                with open(detfolder + '/' + segmentData.foldername + '/' + frameName,'a+') as det:
                    for i,gtData in gtDF.iterrows():
                        if segmentData.left <= gtData.xcentre and segmentData.top <= gtData.ycentre and \
                            segmentData.right > gtData.xcentre and segmentData.bottom > gtData.ycentre:
                            gt.write('{} {} {} {} {}\n'.format(gtData['class'],gtData.left,gtData.top,gtData.right,gtData.bottom))
                            
                            nearestDet = ''
                            
                            for k,detData in detDF.iterrows():
                                #find the nearest detection centre to groundtruth centre
                                gtDistance = np.sqrt(np.square(detData.xcentre - gtData.xcentre) + \
                                                    np.square(detData.ycentre - gtData.ycentre))
#                                print('nearest',type(nearestDet))
#                                print(len(nearestDet))
                                if len(nearestDet) == len(''):
                                    nearestDet = detData
                                    nearestDet['gtDist'] = gtDistance
                                elif nearestDet['gtDist'] < gtDistance:
                                    nearestDet = detData
                                    nearestDet['gtDist'] = gtDistance
                            if len(nearestDet) != len(''):
                                # check if the nearestDetection is close enough
                                # check if detection should be in this segment
                                
                                
                                # if any of the above is true, detection belongs
                                # to this segment
                                nearestDetSeg = False # assume that this is not the segment for the nearest detection
                                if nearestDet['gtDist'] <= dthreshold:
                                    det.write('{} {} {} {} {} {}\n'.format(nearestDet['class'],nearestDet.conf,nearestDet.left,nearestDet.top,nearestDet.right,nearestDet.bottom))
                                    nearestDetSeg = True
                                else:#check if nearest detection is close enough to other groundtruth values
                                    #check if nearestDet should be in current segment
                                    nearestDetSeg = (segmentData.left <= nearestDet.xcentre) and (segmentData.top <= nearestDet.ycentre) and \
                                                (segmentData.right > nearestDet.xcentre) and (segmentData.bottom > nearestDet.ycentre)
                                    if nearestDetSeg:
                                        #if it should be in this segment, check if it is
                                        #close enough to another groundtruth value
                                        
                                        for i2,gtData2 in gtDF.iterrows():
                                            #if this groundtruth is in current segment, then do not write it
                                            #cos outer looping through groundtruth will handle it
#                                            gt2Seg = segmentData.left <= gtData2.xcentre and segmentData.top <= gtData2.ycentre and \
#                                                segmentData.right > gtData2.xcentre and segmentData.bottom > gtData2.ycentre
#                                            if gt2Seg:
#                                                continue
                                            #find distance between nearestDet and gtData2
                                            nearestGTdist = np.sqrt(np.square(nearestDet.xcentre - gtData2.xcentre) + \
                                                        np.square(nearestDet.ycentre - gtData2.ycentre))
                                            
                                            if nearestGTdist <= dthreshold:
                                                #found out that nearestDet is close enough to another ground truth
                                                # do not include nearestDet in this segment
                                                # exit loop
                                                nearestDetSeg = False
                                                break
                                            
                                    if nearestDetSeg:
#                                        print(nearestDet)
#                                        print(segmentData)
#                                        print(gtData)
#                                        input('>')
#                                        print(frameName + ': {} {} {} {} {} {}'.format(nearestDet['class'],nearestDet.conf,nearestDet.left,nearestDet.top,nearestDet.right,nearestDet.bottom))
                                        FP = True
                                        det.write('{} {} {} {} {} {}\n'.format(nearestDet['class'],nearestDet.conf,nearestDet.left,nearestDet.top,nearestDet.right,nearestDet.bottom))
                                        
                                    
                                        
                                        
                                    
#                            if abs(detData.xcentre - gtData.xcentre) <= dthreshold and \
#                                abs(detData.ycentre - gtData.ycentre) <= dthreshold:
#                                    det.write('{} {} {} {} {} {}\n'.format(detData['class'],detData.conf,detData.left,detData.top,detData.right,detData.bottom))
                                    if nearestDetSeg:#if data was written to this segment for nearestDet
                                        #find the maximum distance
                                        if maxDetcxy == 'nan':
                                            maxDetcxy = [abs(nearestDet.xcentre - gtData.xcentre),abs(nearestDet.ycentre - gtData.ycentre)]
                                        else:
                                            if maxDetcxy[0] < abs(nearestDet.xcentre - gtData.xcentre):
                                                maxDetcxy[0] = abs(nearestDet.xcentre - gtData.xcentre)
                                            
                                            if maxDetcxy[1] < abs(nearestDet.ycentre - gtData.ycentre):
                                                maxDetcxy[1] = abs(nearestDet.ycentre - gtData.ycentre)
            if FP:
                print(fidx,frameName,maxDetcxy)
                gtfile = gtfolder + '/' + segmentData.foldername + '/' + frameName
                detfile = detfolder + '/' + segmentData.foldername + '/' + frameName
                drawBoxes(gtfile,detfile,(0,0,960,540))
                input('>')
        if maxDetcxy != 'nan' and (maxDetcxy[0] > dthreshold or maxDetcxy[1] > dthreshold):
            print(fidx,frameName,maxDetcxy)
            print()
                                        
#            segment = segmentsDF.loc[(segmentsDF['left'] <= rowData['xcentre']) & \
#                                    (segmentsDF['top'] <= rowData['ycentre']) & \
#                                    (segmentsDF['right'] > rowData['xcentre']) & \
#                                    (segmentsDF['bottom'] > rowData['ycentre']),:]
            
        
#        return  gtDF,detDF,gtfiles
#    return gtfiles
#segmentsDF = generateSegments(nrows,ncols,frameHeight,frameWidth,gtfolder,detfolder)
#gtDF,detDF,gtfiles= \
segment_gt_n_detections(pfolder,gtfolder,detfolder,frameWidth,frameHeight,nrows,ncols,dthreshold)