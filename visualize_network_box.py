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
parentDIR = '/home/elcymon/litter-detection/darknet/videos/2019011GOPR9027frames'
dirs=['0292-v4-yolov3-tiny-litter_10000-th0p0-nms0p0-iSz416',
    '0292-v4-yolov3-tiny-litter_10000-th0p0-nms0p0-iSz116',
    '0292-v4-yolov3-tiny-litter_10000-th0p0-nms0p0-iSz216',
    '0292-v4-yolov3-litter_10000-th0p5-nms0p0-iSz608']
txtDir = '0_0-960_540'
ntwkBox = pd.DataFrame(index=('xc','yc','w','h'),columns=('116','216','416'))
imgBox = pd.DataFrame(index=('l','t','r','b'),columns=('116','216','416'))

ntwkBox = {}
imgBox = {}
colors = {'116':'r','216':'g','416':'b','GT':'k'}
for d in dirs:
    print(d)
    ntwkDim = d[-3:]
    # print(glob(parentDIR + '/' + d + '/' + txtDir + '/*.txt'))
    for f in glob(parentDIR + '/' + d + '/' + txtDir + '/*.txt'):
        txtDF = pd.read_csv(f,sep=' ', names=['class','conf','x1','y1','x2','y2'])

        fname = ntpath.basename(f)
        if 'ntwk' in fname:
            if ntwkDim not in ntwkBox:
                ntwkBox[ntwkDim] = txtDF.loc[:,'x1':]
            else:
                print('this should not occur for ntwk')
        else:
            if ntwkDim not in imgBox:
                imgBox[ntwkDim] = txtDF.loc[:,'x1':]
            else:
                print('Are you sure about this')

def drawBoxes(boxDict,name,xmin=0,xmax=1,ymin=0,ymax=1):
    fig,ax = plt.subplots()
    for data in ['116','216','416','608']:
        for i,r in boxDict[data].iterrows():
            # print(r)
            if xmax == 1:
                xc,yc,w,h = r.x1,r.y1,r.x2,r.y2
                lx = xc - w / 2.0
                by = 1 - yc - h / 2.0
            else:
                lx = r.x1
                by = 540 - r.y2
                w = r.x2 - r.x1
                h = r.y2 - r.y1

            if(i % 4 == 0):
                if data == '608':
                    data = 'GT'
                rect = mpl.patches.Rectangle((lx,by),w,h,linewidth=2,edgecolor=colors[data],facecolor='none',label=data)
            else:
                rect = mpl.patches.Rectangle((lx,by),w,h,linewidth=2,edgecolor=colors[data],facecolor='none')
            ax.add_patch(rect)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.legend()

    fig.savefig(name,bbox_inches='tight')
    plt.show()
        # print(data)
drawBoxes(ntwkBox,parentDIR + '/0292-ntwkOutput.pdf')
drawBoxes(imgBox,parentDIR + '/0292-imgOutput.pdf',xmin=0,xmax=960,ymin=0,ymax=540)