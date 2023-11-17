#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sb
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

from scipy import stats
import matplotlib.pyplot as plt
import os.path as path
plt.style.use('seaborn-whitegrid')


# In[12]:

class RankingPlugin:
 def input(self, inputfile):
    self.infile = inputfile
    self.df_all = pd.read_csv(inputfile, header=None)
 def run(self):
     pass
 def output(self, outputfile):
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  self.df_all = self.df_all[(self.df_all.algorithm != 'alecar6')]

  algorithms = self.df_all['algorithm'].unique()

  print(len(self.df_all))
  num_traces = len(self.df_all)

  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces
  #print(self.df_all)

  sorted_df = self.df_all.sort_values(['dataset', 'cache_size', 'traces', 'hit_rate'], ascending=[True, True, True, False])
  sorted_df['rank'] = sorted_df.groupby(['dataset', 'cache_size', 'traces'])['hit_rate'].rank(ascending=False, method='dense')
  #print(sorted_df)
  reranked = []
  grouped = sorted_df.groupby(['dataset', 'cache_size', 'traces'])
  for name, group in grouped:
    #print(group)
    top_hit = group.iloc[0]['hit_rate'].item()
    five_p_less = top_hit * 0.95
    #print(five_p_less)
    for index, row in group.iterrows():
        #print(row)
        if row['hit_rate'] >= five_p_less:
            row['rank'] = 1
        reranked.append(row)

  reranked_df = pd.DataFrame(reranked)
  reranked_df.columns = ['traces','algorithm','cache_size','hit_rate', 'dataset', 'rank']
  #print(reranked_df)
  #reranked_df.to_csv('reranked.csv', index=False)
  self.df_all_grouped = reranked_df.groupby(['traces', 'cache_size'])

  top_algos = dict.fromkeys(algorithms, 0)
  second_algos = dict.fromkeys(algorithms, 0)

  scan_diff_from_second = []
  scan_diff_from_top = []
  trace = []
  cache = []
  scan = []
  top = []
  scan_second_count = 0
  divider = 0
  print('len of group: ' + str(len(self.df_all_grouped)))
  for name, group in self.df_all_grouped:
    #print(group)
    
    top_row = group.loc[group['rank']==1]
    #if (top_row.iloc[0]['cache_size'] == 0.05):
    divider += len(top_row)
    #print (top_row)
    for x in top_row['algorithm']:
        top_algos[x] += 1

    #if 'scanalecar' in [x for x in top_row['algorithm']]:
    #    scan_hit = top_row.iloc[0]['hit_rate']
        
    #    scan_diff_from_second.append(scan_hit - second_hit)
    #else:
#         scan_row = group.loc[group['algorithm'] == 'scanalecar']
#         scan_hit = scan_row.iloc[0]['hit_rate']
#         top_hit = top_row.iloc[0]['hit_rate']
#         cache.append(scan_row['cache_size'].item())
#         trace.append(scan_row['traces'].item())
#         scan_diff_from_top.append(top_hit - scan_hit)
#         scan.append(scan_hit)
#         top.append(top_hit)


# np_top = np.asarray(top)
# np_scan_diff_from_top = np.asarray(scan_diff_from_top)
# np_scan = np.asarray(scan)
# diff_percent = (np_scan_diff_from_top/np_top) *100

#print('min scan hit rate: {:.6}'.format(np_scan.min()))
#print('len when scan is top: {}'.format(len(scan_diff_from_second)))
#print('len when scan is not top: {}'.format(len(scan_diff_from_top)))
#print('sum for validation: {}'.format(len(scan_diff_from_second)+len(scan_diff_from_top)))

#print(np_scan_diff_from_top.max())
#print(np_top[np_scan_diff_from_top.argmax()])
#print(np_scan[np_scan_diff_from_top.argmax()])
#worst_case = (np_scan_diff_from_top.max()/np_top[np_scan_diff_from_top.argmax()]) * 100

#print('Scanalecar in worst case is within {:.9}% range of the best performer'.format(worst_case))
#print('Average worst case: {:.3}% with std: {:.3}'.format(np.average(diff_percent), np.std(diff_percent)))

#print('worst case difference: {:.6}% for trace: {} and cache size: {}'
#      .format(diff_percent.max(), trace[diff_percent.argmax()], cache[diff_percent.argmax()]))

  normalized_list_values=[]
  unnormalized_list_values=[]
  x_axis=[]
  secondtop_val= []
  print('normalized')
  for key, val in top_algos.items():
    print('{}: {:.2f}%'.format(key, val/divider * 100))
    normalized_list_values.append(val/divider * 100)
    x_axis.append(key)

  print('unnormalized')
  for key, val in top_algos.items():
    print('{}: {:.2f}%'.format(key, val/len(self.df_all_grouped) * 100))
    unnormalized_list_values.append(val/len(self.df_all_grouped) * 100)
    
  for key, val in second_algos.items():
    secondtop_val.append(val/divider * 100)
    print('{}: {:.2f}%'.format(key, val/divider * 100))
    


  # In[8]:


  #25.8 + 22.3
  (33.89 - 20.99)/33.89


# scanalecar is doing best in 21.99% of the traces  
# 
# 
# #### Percent of traces where each algorithm performs best including ALeCaR in the comparison
# scanalecar: 21.99%  
# dlirs: 15.64%  
# arc: 15.70%  
# lirs: 19.92%  
# lecar: 12.52%   
# alecar6: 7.84%   
# lfu: 3.48%  
# lru: 2.92%    
# 
# 
# #### Percent of traces where each algorithm performs best without ALeCaR 
# scanalecar: 23.90%  
# dlirs: 16.95%  
# arc: 17.04%  
# lirs: 21.58%  
# lecar: 13.60%     
# lfu: 3.76%  
# lru: 3.17% 
# 
# 

# #### Comparison with normalized data

# In[13]:


  fig, ax2 = plt.subplots(figsize=(8, 6))
  ax2 = sns.barplot(x_axis, normalized_list_values)
  plt.ylim(0, 40)
  for x in ax2.patches:
    height = x.get_height()
    ax2.text(x.get_x() + x.get_width()/2, height + 0.5,'{:1.2f}%'.format(height), color='black', ha="center")
  ax2.set_xlabel("Algorithm",  fontsize=16)
  ax2.set_ylabel("Percentage top rank",  fontsize=16)

  fig.savefig('all_five_perc_with_alecar_0.05.png', format='png', bbox_inches = 'tight', dpi=600)
  plt.show()


# #### Comparison with unnormalized data

# In[14]:


  ax2 = sns.barplot(x_axis, unnormalized_list_values)
  plt.ylim(0, 100)
  for x in ax2.patches:
    height = x.get_height()
    ax2.text(x.get_x() + x.get_width()/2, height + 0.5,'{:1.2f}%'.format(height), color='black', ha="center")


  # In[15]:


  self.df_all = pd.read_csv(self.infile, header=None)
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  self.df_all = self.df_all[(self.df_all.algorithm != 'alecar6')]

  algorithms = self.df_all['algorithm'].unique()

  print(len(self.df_all))
  num_traces = len(self.df_all)

  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces
  #print(self.df_all)

  sorted_df = self.df_all.sort_values(['dataset', 'cache_size', 'traces', 'hit_rate'], ascending=[True, True, True, False])
  sorted_df['rank'] = sorted_df.groupby(['dataset', 'cache_size', 'traces'])['hit_rate'].rank(ascending=False, method='dense')
  #print(sorted_df)
  reranked = []
  grouped = sorted_df.groupby(['dataset', 'cache_size', 'traces'])
  for name, group in grouped:
    #print(group)
    top_hit = group.iloc[0]['hit_rate'].item()
    five_p_less = top_hit * 0.95
    #print(five_p_less)
    for index, row in group.iterrows():
        #print(row)
        if row['hit_rate'] >= five_p_less:
            row['rank'] = 1
        reranked.append(row)

  reranked_df = pd.DataFrame(reranked)
  reranked_df.columns = ['traces','algorithm','cache_size','hit_rate', 'dataset', 'rank']
  #print(reranked_df)
  #reranked_df.to_csv('reranked.csv', index=False)
  self.df_all_grouped = reranked_df.groupby(['traces', 'cache_size'])

  top_algos = dict.fromkeys(algorithms, 0)
  second_algos = dict.fromkeys(algorithms, 0)

  scan_diff_from_second = []
  scan_diff_from_top = []
  trace = []
  cache = []
  scan = []
  top = []
  scan_second_count = 0
  divider = 0
  print('len of group: ' + str(len(self.df_all_grouped)))
  for name, group in self.df_all_grouped:
    #print(group)
    
    top_row = group.loc[group['rank']==1]
    if (top_row.iloc[0]['cache_size'] == 0.0005):
        divider += len(top_row)
    #print (top_row)
        for x in top_row['algorithm']:
            top_algos[x] += 1

    #if 'scanalecar' in [x for x in top_row['algorithm']]:
    #    scan_hit = top_row.iloc[0]['hit_rate']
        
    #    scan_diff_from_second.append(scan_hit - second_hit)
    #else:
#         scan_row = group.loc[group['algorithm'] == 'scanalecar']
#         scan_hit = scan_row.iloc[0]['hit_rate']
#         top_hit = top_row.iloc[0]['hit_rate']
#         cache.append(scan_row['cache_size'].item())
#         trace.append(scan_row['traces'].item())
#         scan_diff_from_top.append(top_hit - scan_hit)
#         scan.append(scan_hit)
#         top.append(top_hit)


# np_top = np.asarray(top)
# np_scan_diff_from_top = np.asarray(scan_diff_from_top)
# np_scan = np.asarray(scan)
# diff_percent = (np_scan_diff_from_top/np_top) *100

#print('min scan hit rate: {:.6}'.format(np_scan.min()))
#print('len when scan is top: {}'.format(len(scan_diff_from_second)))
#print('len when scan is not top: {}'.format(len(scan_diff_from_top)))
#print('sum for validation: {}'.format(len(scan_diff_from_second)+len(scan_diff_from_top)))

#print(np_scan_diff_from_top.max())
#print(np_top[np_scan_diff_from_top.argmax()])
#print(np_scan[np_scan_diff_from_top.argmax()])
#worst_case = (np_scan_diff_from_top.max()/np_top[np_scan_diff_from_top.argmax()]) * 100

#print('Scanalecar in worst case is within {:.9}% range of the best performer'.format(worst_case))
#print('Average worst case: {:.3}% with std: {:.3}'.format(np.average(diff_percent), np.std(diff_percent)))

#print('worst case difference: {:.6}% for trace: {} and cache size: {}'
#      .format(diff_percent.max(), trace[diff_percent.argmax()], cache[diff_percent.argmax()]))

  normalized_list_values=[]
  unnormalized_list_values=[]
  x_axis=[]
  secondtop_val= []
  print('normalized')
  for key, val in top_algos.items():
    print('{}: {:.2f}%'.format(key, val/divider * 100))
    normalized_list_values.append(val/divider * 100)
    x_axis.append(key)

  print('unnormalized')
  for key, val in top_algos.items():
    print('{}: {:.2f}%'.format(key, val/len(self.df_all_grouped) * 100))
    unnormalized_list_values.append(val/len(self.df_all_grouped) * 100)
    
  for key, val in second_algos.items():
    secondtop_val.append(val/divider * 100)
    print('{}: {:.2f}%'.format(key, val/divider * 100))


  # In[16]:


  fig, ax2 = plt.subplots(figsize=(8, 6))
  ax2 = sns.barplot(x_axis, normalized_list_values)
  plt.ylim(0, 40)
  for x in ax2.patches:
    height = x.get_height()
    ax2.text(x.get_x() + x.get_width()/2, height + 0.5,'{:1.2f}%'.format(height), color='black', ha="center")
  ax2.set_xlabel("Algorithm",  fontsize=16)
  ax2.set_ylabel("Percentage top rank",  fontsize=16)

  fig.savefig('all_five_perc_with_alecar_0.05.png', format='png', bbox_inches = 'tight', dpi=600)
  plt.show()


  # In[17]:


  self.df_all = pd.read_csv(self.infile, header=None)
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate']

  self.df_all = self.df_all[(self.df_all.algorithm != 'alecar6')]

  algorithms = self.df_all['algorithm'].unique()

  print(len(self.df_all))
  num_traces = len(self.df_all)
 
  labels = list(r[-1] for r in self.df_all['traces'].str.split('/'))

  ls = []
  trunk_traces = []
  for i in labels:
    trunk_traces.append(path.splitext(i)[0])
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('vps'):
        ls.append('CloudVPS')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('webserver'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blk' and path.splitext(i)[0].startswith('moodle'):
        ls.append('CloudCache')
    if path.splitext(i)[-1] == '.blkparse':
        ls.append('FIU')
    if path.splitext(i)[-1]== '.csv':
        ls.append('MSR')
    if path.splitext(i)[-1]== '.txt':
        ls.append('NEXUS')
  self.df_all['dataset'] = ls
  self.df_all['traces'] = trunk_traces
  #print(self.df_all)

  sorted_df = self.df_all.sort_values(['dataset', 'cache_size', 'traces', 'hit_rate'], ascending=[True, True, True, False])
  sorted_df['rank'] = sorted_df.groupby(['dataset', 'cache_size', 'traces'])['hit_rate'].rank(ascending=False, method='dense')
  #print(sorted_df)
  reranked = []
  grouped = sorted_df.groupby(['dataset', 'cache_size', 'traces'])
  for name, group in grouped:
    #print(group)
    top_hit = group.iloc[0]['hit_rate'].item()
    five_p_less = top_hit * 0.95
    #print(five_p_less)
    for index, row in group.iterrows():
        #print(row)
        if row['hit_rate'] >= five_p_less:
            row['rank'] = 1
        reranked.append(row)

  reranked_df = pd.DataFrame(reranked)
  reranked_df.columns = ['traces','algorithm','cache_size','hit_rate', 'dataset', 'rank']
  #print(reranked_df)
  #reranked_df.to_csv('reranked.csv', index=False)
  self.df_all_grouped = reranked_df.groupby(['traces', 'cache_size'])

  top_algos = dict.fromkeys(algorithms, 0)
  second_algos = dict.fromkeys(algorithms, 0)

  scan_diff_from_second = []
  scan_diff_from_top = []
  trace = []
  cache = []
  scan = []
  top = []
  scan_second_count = 0
  divider = 0
  print('len of group: ' + str(len(self.df_all_grouped)))
  for name, group in self.df_all_grouped:
    #print(group)
    
    top_row = group.loc[group['rank']==1]
    if (top_row.iloc[0]['cache_size'] == 0.001):
        divider += len(top_row)
    #print (top_row)
        for x in top_row['algorithm']:
            top_algos[x] += 1

    #if 'scanalecar' in [x for x in top_row['algorithm']]:
    #    scan_hit = top_row.iloc[0]['hit_rate']
        
    #    scan_diff_from_second.append(scan_hit - second_hit)
    #else:
#         scan_row = group.loc[group['algorithm'] == 'scanalecar']
#         scan_hit = scan_row.iloc[0]['hit_rate']
#         top_hit = top_row.iloc[0]['hit_rate']
#         cache.append(scan_row['cache_size'].item())
#         trace.append(scan_row['traces'].item())
#         scan_diff_from_top.append(top_hit - scan_hit)
#         scan.append(scan_hit)
#         top.append(top_hit)


# np_top = np.asarray(top)
# np_scan_diff_from_top = np.asarray(scan_diff_from_top)
# np_scan = np.asarray(scan)
# diff_percent = (np_scan_diff_from_top/np_top) *100

#print('min scan hit rate: {:.6}'.format(np_scan.min()))
#print('len when scan is top: {}'.format(len(scan_diff_from_second)))
#print('len when scan is not top: {}'.format(len(scan_diff_from_top)))
#print('sum for validation: {}'.format(len(scan_diff_from_second)+len(scan_diff_from_top)))

#print(np_scan_diff_from_top.max())
#print(np_top[np_scan_diff_from_top.argmax()])
#print(np_scan[np_scan_diff_from_top.argmax()])
#worst_case = (np_scan_diff_from_top.max()/np_top[np_scan_diff_from_top.argmax()]) * 100

#print('Scanalecar in worst case is within {:.9}% range of the best performer'.format(worst_case))
#print('Average worst case: {:.3}% with std: {:.3}'.format(np.average(diff_percent), np.std(diff_percent)))

#print('worst case difference: {:.6}% for trace: {} and cache size: {}'
#      .format(diff_percent.max(), trace[diff_percent.argmax()], cache[diff_percent.argmax()]))

  normalized_list_values=[]
  unnormalized_list_values=[]
  x_axis=[]
  secondtop_val= []
  print('normalized')
  for key, val in top_algos.items():
    print('{}: {:.2f}%'.format(key, val/divider * 100))
    normalized_list_values.append(val/divider * 100)
    x_axis.append(key)

  print('unnormalized')
  for key, val in top_algos.items():
    print('{}: {:.2f}%'.format(key, val/len(self.df_all_grouped) * 100))
    unnormalized_list_values.append(val/len(self.df_all_grouped) * 100)
    
  for key, val in second_algos.items():
    secondtop_val.append(val/divider * 100)
    print('{}: {:.2f}%'.format(key, val/divider * 100))


  # In[18]:


  fig, ax2 = plt.subplots(figsize=(8, 6))
  ax2 = sns.barplot(x_axis, normalized_list_values)
  plt.ylim(0, 40)
  for x in ax2.patches:
    height = x.get_height()
    ax2.text(x.get_x() + x.get_width()/2, height + 0.5,'{:1.2f}%'.format(height), color='black', ha="center")
  ax2.set_xlabel("Algorithm",  fontsize=16)
  ax2.set_ylabel("Percentage top rank",  fontsize=16)

  fig.savefig(outputfile, format='png', bbox_inches = 'tight', dpi=600)
  plt.show()


  # In[ ]:




