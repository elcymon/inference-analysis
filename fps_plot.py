import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rc('image',cmap='plasma')

#from glob import glob
#import numpy as np
import matplotlib.pyplot as plt

#import pathlib
import os
fpath = ['..','darknet','videos'] # folder path to csv file
fname = '20190111GOPR9027qtr-yolov3-tiny-litter_10000-th0p1-nms1p0.csv' # name of csv file

def fps_plot(fpathList,fname,platform='unknown'):
    resultName = os.sep.join(fpathList + [platform + '-' + fname[:-4]])
    
    filePath = os.sep.join(fpathList + [fname])
    df = pd.read_csv(filePath,sep=',',engine='python')
    df['loopDuration'] = df['loopDuration'] * 1000 # convert loop duration to ms
    
    f = plt.figure()
    ax = f.gca()
    
    df.plot(x='Time',y=['loopDuration','InferenceTime'],cmap= mpl.cm.plasma,ax=ax)
    ax.set_ylabel('Time in milliseconds',fontsize=16,fontweight='bold')
    ax.set_xlabel('Time in seconds',fontsize=16,fontweight='bold')
    ax.tick_params(axis='both',which='major',labelsize=16)
    ax.legend(fontsize=16,loc='upper center', ncol=2,bbox_to_anchor=(0.5,1.2))
    ax.set_xlim([0, df['Time'].max() * 1.1])
    ax.set_ylim([0, df['loopDuration'].max() * 1.1])
    
    summaryCols = ['loopDuration','InferenceTime']
    meanSTD = pd.DataFrame(index=['Mean','Standard Deviation'],columns=summaryCols)
    meanSTD.loc['Mean',['loopDuration','InferenceTime']] = df[summaryCols].mean()
    meanSTD.loc['Standard Deviation',summaryCols] = df[summaryCols].std()
    f.savefig(resultName + '.pdf',bbox_inches='tight')
    meanSTD.to_csv(resultName + '.csv')
    return df,meanSTD