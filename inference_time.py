import matplotlib
from glob import glob
import pandas as pd
import ntpath
import copy
import numpy as np
def create_multindex_df(platforms,videoquality,dnn_network):
    columns = []
    all_data_cols = []
    for p in platforms:
        for dnn in dnn_network.keys():
            for sz in dnn_network[dnn]:
                columns.append(tuple((dnn,sz)))
                for vq in videoquality:
                    all_data_cols.append(tuple((p,vq,f'{dnn}-{sz}')))
    
    
    columns = pd.MultiIndex.from_tuples(columns)
    index = pd.MultiIndex.from_product([platforms,videoquality])
    all_data_cols = pd.MultiIndex.from_tuples(all_data_cols)
    return pd.DataFrame(index = index, columns = columns), pd.DataFrame(columns=all_data_cols)
    

def get_file_index(filename):
    dnn = 'yolo-tiny'
    sz = 128
    platform = 'pi4'
    quality = '1920x1080'
    if 'mobilenetSSD' in filename:
        dnn = 'mobilenetSSD'
        if '124' in filename:
            sz = 124
        else:
            sz = 220
    elif ('yolov3-tiny' in filename) and ('224' in filename):
        sz = 224
    if 'half' in filename:
        quality = '960x540'
    if 'erlebrain' in filename:
        platform = 'erlebrain'
    return platform,quality,dnn,sz
    
def inference_time():
    platforms = ['pi4','erlebrain']
    videoquality = ['1920x1080','960x540']
    dnn_network = {'mobilenetSSD':[124,220],'yolo-tiny':[128,224]}
    
    resultPath = '../*-inference-time'
    results_df,all_data_df =  create_multindex_df(platforms,videoquality,dnn_network)
    for f in glob(resultPath+ "/*.csv"):
        print(ntpath.basename(f))
        platform,quality,dnn,sz = get_file_index(f)
        df = pd.read_csv(f,index_col=0,header=0)
        df['inference_tsecs'] = df['InferenceTime'].div(1000)
        mean_t = df['inference_tsecs'].mean()
        std_t = df['inference_tsecs'].std()
        results_df.loc[(platform,quality),(dnn,sz)] = f'${mean_t:.4f}\pm {std_t:.4f}$'
        all_data_df[(platform,quality,f'{dnn}-{sz}')] = df['inference_tsecs']
    cmap =  copy.deepcopy(matplotlib.cm.get_cmap('inferno'))
    cmap = matplotlib.colors.ListedColormap(cmap(np.linspace(0,0.8,256)))
    for p in platforms:
        for vq in videoquality:
            fig = matplotlib.pyplot.figure(figsize=(5,5))
            ax = fig.gca()
            all_data_df[(p,vq)].plot(ax=ax,style='.',cmap=cmap,alpha=1)
            matplotlib.pyplot.legend(ncol=2,loc='upper center',bbox_to_anchor=(0.5,1.15))
            
            ax.set_ylim([0, all_data_df[(p,vq)].max().max() * 1.3])
            ax.set_ylabel('Time in seconds',fontweight='bold')
            ax.set_xlabel('Frame',fontweight='bold')
            fig.savefig(f"../{p}-inference-time/{vq}.png",bbox_inches='tight',dpi=100)
        results_df.to_latex(f"../{p}-inference-time/pi4-erlebrain.tex",escape=False)
    return results_df,all_data_df