import pandas as pd
from glob import glob
filesPath = '../videos/litter-recording/*yolov3-litter*'#/0_0-960_540/*.txt'
#print(len(glob(filesPath)))
""" 
df = pd.read_csv('../GOPR9075-00001.txt',sep=' ',names=['object','conf','left','top','right','bottom'],index_col=None)
print(df)
df.drop(columns='conf',inplace=True)
print(df)
df.to_csv('../GOPR9075-00001-2.txt',sep=' ',header=False,index=False) """

for i,f in enumerate(glob(filesPath)):
    
#    print(f"{f}\t{len(glob(f + '/0_0-960_540/*.txt'))}")
    for i2,f2 in enumerate(glob(f + '/0_0-960_540/*.txt')):
        #print(f2)
        try:
            df = pd.read_csv(f2,sep=' ',names=None,header=None,index_col=None)
        except :
            continue
        df.dropna(axis=1,inplace=True)
        columns = df.shape[1]
        break
    print(f'{i+1}: cols={columns},folder={f},\n{df.head(2)}')
"""         df.drop(columns='conf',inplace=True)
        df.to_csv(f2,sep=' ',header=False,index=False) """
     #   print(len(f2))
