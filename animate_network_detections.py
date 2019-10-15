# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

def animateDetections(df,frameNames,xlim=(0,960), ylim=(0,540)):
    colors = ['b','g','r','k']#plt.cm.inferno(np.linspace(0.2,0.9,4))
    fig = plt.figure(figsize=(5,3))
#    plt.tight_layout(pad=0)
    
    
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, \
                         xlim=xlim,ylim=ylim)
    fig.set_tight_layout('tight')
#    gttp = drawBBox((0,0,0,0),'none')
#    
#    dettp = drawBBox((0,0,0,0),'none')
#    
#    fp = drawBBox((0,0,0,0),'none')
#    
#    fn = drawBBox((0,0,0,0),'none')
    def drawBBox(bbox,boxColor,label):
        x1,y1,x2,y2 = bbox
        ymin,ymax = ylim
        rect = plt.Rectangle((x1,ymax - y2), x2-x1, y2-y1,\
                             ec=boxColor,lw=1,fc='none', label = label)
        ax.add_patch(rect)
        
        return rect
    
    
    def removePatches():
        for p in reversed(ax.patches):
            p.remove()
    
    
    def init():
        gttp = drawBBox((0,0,0,0),colors[0],'GT')
        dettp = drawBBox((0,0,0,0),colors[2],'TP')
        fp = drawBBox((0,0,0,0),colors[3],'FP')
        fn = drawBBox((0,0,0,0),colors[1],'FN')
        plt.legend(fontsize=10,ncol=4,loc='upper center', bbox_to_anchor=(0.5,1.15))
        
        return gttp,dettp,fp,fn
    
    def convertToBBox(data):
        if data is not np.nan:
            data = eval(data)
        else:
            data = None
        return data
    
    def animationStep(imageName):
        print(imageName)
        frameDF = df.loc[df['imageName'] == imageName, :]
        
        removePatches()
        
        gttp = drawBBox((0,0,0,0),'none','GT')
        dettp = drawBBox((0,0,0,0),'none','TP')
        fp = drawBBox((0,0,0,0),'none','FP')
        fn = drawBBox((0,0,0,0),'none','FN')
        
        for i in frameDF.index:
            gtBBox =  convertToBBox(frameDF.loc[i,'GTBBox'])
            detBBox = convertToBBox(frameDF.loc[i,'DetBBox'])
            if int(frameDF.loc[i,'TP']) == 1:
                # true positive
                gttp = drawBBox(gtBBox,colors[0], 'GT')
                dettp = drawBBox(detBBox,colors[2], 'TP')
            elif int(frameDF.loc[i,'FP']) == 1:
                # false positive
                fp = drawBBox(detBBox,colors[3], 'FP')
            elif int(frameDF.loc[i,'FN']) == 1:
                # false negative
                fn = drawBBox(gtBBox,colors[1], 'FN')
            else:
                print('should not happen')
        
        return gttp,dettp,fp,fn
    ani = animation.FuncAnimation(fig,animationStep,frames=frameNames,blit=True,init_func=init)
    plt.close()
    return ani

def loadCSV(fileName):
    df = pd.read_csv(fileName,index_col=0)
    frameNames = sorted(list(set(df['imageName'])))
    return df,frameNames

if __name__ == '__main__':
    network = ['GOPR9027-mobilenet-124',\
               'GOPR9027-mobilenet-220',\
               'GOPR9027-yolov3-tiny-128',\
               'GOPR9027-yolov3-tiny-224']
    networksLogDF = pd.DataFrame(columns=['TP','FP','FN'])
    resultPath = '../videos/litter-recording/' 
    for n in network:
        fileName = resultPath + n +'/1r1c/analysis/logInfoDF.csv'
        df,frameNames = loadCSV(fileName)
        
        networksLogDF.loc[n.replace('GOPR9027-',''),:] = list(df.loc[:,'TP':'FN'].sum())
        
        
#        ani = animateDetections(df,frameNames)
#        ani.save(fileName.replace('logInfoDF.csv',n + '.mp4'), fps=50, dpi = 300)
#        plt.close()
    networksLogDF.astype(int).to_latex(resultPath + 'GOPR9027.tex')