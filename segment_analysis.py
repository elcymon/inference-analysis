# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dname = '20190111GOPR9027half-yolov3-tiny-litter_10000_216-th0p1-nms0p0-iSz216-seg18r32c'
fname='../darknet/videos/' + dname + '/segmentsMAP.csv'

df = pd.read_csv(fname,sep=',|_|-',engine='python',names=['left','top','right','bottom','mAP'])
leftRight = df.apply(lambda x: (int(x[0]), int(x[2])),axis=1)
leftRight = leftRight.sort_values()
leftRight = leftRight.apply(lambda x: '%s-%s' % x)
leftRight = leftRight.unique().astype(str)

topBottom = df.apply(lambda x: (int(x[1]), int(x[3])),axis=1)
topBottom = topBottom.sort_values()
topBottom = topBottom.apply(lambda x: '%s-%s' % x)
topBottom = topBottom.unique().astype(str)

heatData = pd.DataFrame(columns=leftRight,index=topBottom)

for index,row in df.iterrows():
    lr = '%s-%s' % (int(row.left),int(row.right))
    tb = '%s-%s' % (int(row.top),int(row.bottom))
    mAP = round(row.mAP,2)
    heatData.loc[tb,lr] = mAP

f = plt.figure(figsize=(17,6))
    
ax = sns.heatmap(heatData.astype(float),annot=True,vmin=0,vmax=1,cmap='magma')
ax.xaxis.set_ticks_position('top')
ax.figure.axes[-1].tick_params(axis='both', which='major', labelsize=30)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
f.savefig(fname[:-4] + '.pdf',bbox_inches='tight')
