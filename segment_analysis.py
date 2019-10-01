# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ntpath
from glob import glob
import numpy as np
#vfolder = '20190111GOPR9027half-hflip-mobilenetSSD-10000-th0p5-nms0p0-iSz124'
#vsegments = '3r4c'
#dname = '20190111GOPR9027/20190111GOPR9027half-hflip/' + vfolder + '/0_0-960_540/' + vsegments #'20190111GOPR9027half-mobilenetSSD-10000-th0p5-nms0p0-iSz124-seg3r4c'#'20190111GOPR9027half-yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128-seg3r4c'
#fname='../darknet/videos/' + dname + '/segmentsMAP.csv'
#figname = '../darknet/videos/' + dname + '/' + vfolder + '-' + vsegments
def processFile(csvFile,nRowsmCols,networkFolder,resultsFolder,oneVideo=True):
    if oneVideo and nRowsmCols != '':
        fileFolder = os.sep.join([networkFolder,'0_0-960_540'])
    
    
    if 'detectionCounts' in csvFile:
        fname = glob(os.sep.join([resultsFolder,fileFolder,nRowsmCols,'*' + csvFile]))[0]
        figname = os.sep.join([ntpath.dirname(fname),'-'.join([networkFolder, nRowsmCols,'detectionCounts'])])
        df = pd.read_csv(fname,sep=',|_|-',engine='python',names=['left','top','right','bottom','Count'],skiprows=[0])
    else:
        fname = os.sep.join([resultsFolder,fileFolder,nRowsmCols,csvFile])
        figname = os.sep.join([ntpath.dirname(fname),'-'.join([networkFolder, nRowsmCols,csvFile[:-4]])])
        df = pd.read_csv(fname,sep=',|_|-',engine='python',names=['left','top','right','bottom','mAP'])
    # print(df)
    leftRight = df.apply(lambda x: (int(x[0]), int(x[2])),axis=1)
    leftRight = leftRight.sort_values()
    leftRight = leftRight.apply(lambda x: '%s-%s' % x)
    leftRight = leftRight.unique().astype(str)

    topBottom = df.apply(lambda x: (int(x[1]), int(x[3])),axis=1)
    topBottom = topBottom.sort_values()
    topBottom = topBottom.apply(lambda x: '%s-%s' % x)
    topBottom = topBottom.unique().astype(str)

    heatData = pd.DataFrame(columns=leftRight,index=topBottom)

    for _,row in df.iterrows():
        lr = '%s-%s' % (int(row.left),int(row.right))
        tb = '%s-%s' % (int(row.top),int(row.bottom))
        mAP = round(row.iloc[-1],2)
        heatData.loc[tb,lr] = mAP
    
    return (figname,heatData)

def plotHeatMap(figname,heatData,vmin=0,vmax=1):
    f = plt.figure(figsize=(4,3))#)figsize=(17,6)
        
    ax = sns.heatmap(heatData.astype(float),annot=True,vmin=vmin,vmax=vmax,cmap='magma')
    ax.xaxis.set_ticks_position('top')
    ax.figure.axes[-1].tick_params(axis='both', which='major')#, labelsize=30)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    f.savefig(figname + '.pdf',bbox_inches='tight')
    plt.close()

def analyseVideos(resultsFolder, networkPattern, nRowsmCols='', csvFile='segmentsMAP.csv',  oneVideo=True):
    allData = pd.DataFrame()
    if oneVideo:
        videoCategory = 'oneVideo'
        summaryCSV = networkPattern.replace('*',videoCategory)

    else:
        videoCategory = ''
    networkFolders = glob(resultsFolder + '/' + networkPattern)

    for i,networkFolder in enumerate(networkFolders):
        networkFolder = ntpath.basename(networkFolder)
        print(f'{i+1}/{len(networkFolders)}: {networkFolder}')
        #continue
        figname,heatData = processFile(csvFile,nRowsmCols,networkFolder,resultsFolder,oneVideo=True)
        if 'detectionCounts' in csvFile:
            plotHeatMap(figname,heatData,vmin=0,vmax=heatData.iloc[-1].max())    
        else:
            plotHeatMap(figname,heatData,vmin=0,vmax=1)
        if len(allData) == 0:
            allData = heatData.copy(deep=True)
        else:
            allData += heatData
    
    if csvFile == 'segmentsMAP.csv':
        allData /= len(networkFolders)
    summaryCSV += '-'.join([nRowsmCols,csvFile])
    
    # print(len(networkFolders))
    detectionsCSV = os.sep.join([resultsFolder,summaryCSV])
    allData.to_csv(detectionsCSV)

resultsFolder = '../videos/litter-recording'
networkPattern = '*-yolov3-litter_10000-th0p0-nms0p0-iSz608'
nRowsmCols='5r9c'
oneVideo=True

#analyseVideos(resultsFolder, networkPattern ,nRowsmCols=nRowsmCols,
#                csvFile='segmentsMAP.csv',oneVideo=oneVideo)

analyseVideos(resultsFolder, networkPattern ,nRowsmCols=nRowsmCols,
                csvFile='detectionCounts.csv',oneVideo=oneVideo)
