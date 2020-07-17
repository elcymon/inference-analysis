#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:43:48 2020

@author: elcymon
"""
import ntpath
from glob import glob
import pandas as pd
from astropy.table import Table
folder = '../data/simplified_data/first_and_last_appearance_condition/'
print(folder)
networks = ['mSSD124','mSSD220']
platforms = ['50fps','pi4','erlebrain']

columns = pd.MultiIndex.from_product([platforms,networks])
df = None
fps_list = []
for f in glob(folder + '50fps/*/*TPandFN_metrics_data_nafilter_0.tex') + \
    glob(folder + '*/*TPandFN_metrics_data_nafilter_0.tex'):
    with open(f) as file_obj:
        file_data = [i for i in file_obj.readlines()  \
                     if ('toprule' not in i) and ('midrule' not in i) and
                     ('bottomrule' not in i)]
    
    table = Table.read(file_data,format='latex').to_pandas()
    table.set_index('col0',inplace=True)
    #get fps data
    fps = ntpath.basename(f).split('_')[1].replace('fps','').replace('p','.')
    if '50fps' in f:
        platform = '50fps'
        fps = '50'
    elif 'erlebrain' in f:
        platform = 'erlebrain'
    elif 'pi4' in f:
        platform = 'pi4'
    if df is None:
        df = pd.DataFrame(columns = columns,index = ['fps'] + list(table.index.values))
    
    if 'mSSD-124' in table.columns:
        table.loc['fps','mSSD-124'] = fps
        df[(platform,'mSSD124')] = table.loc[df.index,'mSSD-124']
    if 'mSSD-220' in table.columns:
        table.loc['fps','mSSD-220'] = fps
        df[(platform,'mSSD220')] = table.loc[df.index,'mSSD-220']
    fps_list.append(fps)
    #extract platform and networkname
    
    print(f)

df.to_latex(f'{folder}combined_tp_fn_latex_tables.tex',escape=False)