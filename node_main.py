from default_setting_n import cmd_args
from classifier_based_MIA import train_and_ev_attack_model
from threshold_based_MIA import select_threshold_and_evaluate
from training_process import node_train,get_0_1_2_hop_prediction,compute_metric_of_graph

from torch_geometric.datasets import Reddit,Flickr,CitationFull,Planetoid,LastFM,FacebookPagePage
import pandas as pd
import numpy as np
from datetime import datetime
import os
import torch
import time
from copy import deepcopy
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph,to_networkx
import matplotlib.pyplot as plt
import traceback
# import logging
import math


def explain_attack_model(attack_model,shadow_data,ev_data,dir_nam):
  #explain the prediction here temporarily
  start_time=time.time()
  import shap
  from torch.autograd import Variable
  
  attack_model.eval()
  f=lambda x:attack_model(Variable(torch.from_numpy(x))).detach().numpy()
  x_list=[item for item in ev_data.columns if item != 'label' and item != 'o_index']
  train_data = shadow_data[x_list].to_numpy(dtype=np.float32)
  test_data = ev_data[x_list].to_numpy(dtype=np.float32)#[:10,:]
  explainer = shap.KernelExplainer(f,train_data)#, data
  
  shap_values=explainer.shap_values(test_data)
  print(type(shap_values))
  print(shap_values)

  sum_path=dir_name+'shap_sum_plot_f_i.png' #,features=data
  shap.summary_plot(shap_values=shap_values[0],plot_type='bar',features=test_data,feature_names=x_list,max_display=30,show=False,plot_size=(15,12))
  plt.gcf()
  plt.savefig(sum_path,dpi=150)
  plt.clf()

  beeswarm_path=dir_name+'shap_beeswarm_plot_f_i.png'
  shap.summary_plot(shap_values=shap_values[0],plot_type='dot',features=test_data,feature_names=x_list,max_display=15,show=False,plot_size=(8,6))
  plt.gcf()
  plt.savefig(beeswarm_path,dpi=150)


#get dataset with dataset_name
def get_dataset(dataset_name,root_path):
  #'cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed'
  if dataset_name=='reddit':
    dataset=Reddit(root_path)
  elif dataset_name=='lastfm':
    dataset=LastFM(root_path)
  elif dataset_name=='flickr':
    dataset=Flickr(root_path)
  elif dataset_name=='facebookpagepage':
    dataset=FacebookPagePage(root_path)
  elif dataset_name in ['cora_ml', 'citeseer', 'dblp', 'pubmed']:
    dataset=CitationFull(root=root_path,name=dataset_name)
  elif dataset_name in ['cora']:
    dataset=Planetoid(root=root_path,name=dataset_name,split='random')
  else:
    raise NotImplementedError
  
  return dataset

def compute_shannon_entropy(num_of_class,subgraph_dataset):
  t_dict=subgraph_dataset[0].value_counts().to_dict()
  t_len=subgraph_dataset.shape[0]
  count_list=[]
  print(t_dict)
  print(t_len)
  for class_sign in range(0,num_of_class):
    if class_sign not in t_dict.keys():
      count_list.append(0)
    else:
      count_list.append(t_dict[class_sign])
  H=-sum([(count/t_len)*(math.log2(count/t_len)) for count in count_list if count!=0])
  return round(H/(math.log2(num_of_class)),4)

  

#prepare dataset
def prepare_dataset(target_dataset_name,target_train_test_rate,shadow_dataset_name,shadow_train_test_rate,root_path,target_shadow_rate,sample_way):
  """
  target_shadow_rate (float): The percentage of target dataset while it is in the same dataset with shadow dataset
  """
  #the split for target_train, target_test, shadow_train, and shadow_test https://github.com/iyempissy/rebMIGraph/blob/main/TSTS.py
  len_of_each_class_in_train_data={
    'reddit':500,
    'flickr':1500,
    'cora':90,
    'cora_ml':90,
    'citeseer':100,
    'pubmed':1500,
    'dblp':800,
  }

  if target_dataset_name==shadow_dataset_name:
    #only from one dataset
    total_dataset=get_dataset(target_dataset_name,root_path)
    print(total_dataset.data)
    print(total_dataset.data.x)
    print(total_dataset.data.x.size())
    print(total_dataset.data.edge_index)
    print(total_dataset.data.y)

    num_of_class=torch.max(total_dataset.data.y).item()+1

    #test the feature distribution of total_dataset.data.x
    t_x=deepcopy(total_dataset.data.x.numpy())
    print("min value of feature:%s"%(str(np.min(t_x))))
    print("max value of feature:%s"%(str(np.max(t_x))))

    original_pd=pd.DataFrame(total_dataset.data.y.numpy())
    
    #0: random; 1: all balanced; 2: target_train and shadow_train balanced
    t_pd=pd.DataFrame(total_dataset.data.y.numpy())
    if sample_way==1:
      min_count=((t_pd[0].value_counts().min())//4)*4
      t_pd=t_pd.loc[[item[1] for item in t_pd.groupby(0).apply(pd.DataFrame.sample, n=min_count)[0].index]]
      total_index=[item for item in t_pd.index]

      target_pd=t_pd.loc[[item[1] for item in t_pd.groupby(0).apply(pd.DataFrame.sample, frac=target_shadow_rate)[0].index]]
      target_index=[item for item in target_pd.index]
      target_train=target_pd.loc[[item[1] for item in target_pd.groupby(0).apply(pd.DataFrame.sample, frac=target_train_test_rate)[0].index]]
      target_train_index=[item for item in target_train.index]
      target_test_index=[item for item in target_index if item not in target_train_index]
      target_test=t_pd.loc[target_test_index]

      shadow_index=[item for item in total_index if item not in target_index]
      shadow_pd=t_pd.loc[shadow_index]
      shadow_train=shadow_pd.loc[[item[1] for item in shadow_pd.groupby(0).apply(pd.DataFrame.sample, frac=shadow_train_test_rate)[0].index]]
      shadow_train_index=[item for item in shadow_train.index]
      shadow_test_index=[item for item in shadow_index if item not in shadow_train_index]
      shadow_test=t_pd.loc[shadow_test_index]
    elif sample_way==0:
      total_index=[item for item in t_pd.index]

      target_pd=t_pd.sample(frac=target_shadow_rate,replace = False)
      target_index=[item for item in target_pd.index]
      target_train=target_pd.sample(frac=target_train_test_rate,replace = False)
      target_train_index=[item for item in target_train.index]
      target_test_index=[item for item in target_index if item not in target_train_index]
      target_test=t_pd.loc[target_test_index]

      shadow_index=[item for item in total_index if item not in target_index]
      shadow_pd=t_pd.loc[shadow_index]
      shadow_train=shadow_pd.sample(frac=shadow_train_test_rate,replace=False)
      shadow_train_index=[item for item in shadow_train.index]
      shadow_test_index=[item for item in shadow_index if item not in shadow_train_index]
      shadow_test=t_pd.loc[shadow_test_index]

    elif sample_way==2:
      total_index=[item for item in t_pd.index]
      
      #number for training target and shadow models
      if target_dataset_name in len_of_each_class_in_train_data.keys():
        train_num=len_of_each_class_in_train_data[target_dataset_name]*2
      else:
        raise NotImplementedError

      train_pd=t_pd.loc[[item[1] for item in t_pd.groupby(0).apply(pd.DataFrame.sample, n=train_num)[0].index]]
      train_index=[item for item in train_pd.index]
      target_train=train_pd.loc[[item[1] for item in train_pd.groupby(0).apply(pd.DataFrame.sample, frac=0.5)[0].index]]
      target_train_index=[item for item in target_train.index]
      shadow_train_index=[item for item in train_index if item not in target_train_index]
      shadow_train=t_pd.loc[shadow_train_index]

      test_index=[item for item in total_index if item not in train_index]
      test_pd=t_pd.loc[test_index]
      target_test=test_pd.sample(n=int(num_of_class*train_num/2),replace=False)
      target_test_index=[item for item in target_test.index]
      shadow_test_before_sample_index=[item for item in test_index if item not in target_test_index]
      shadow_test_before_sample=t_pd.loc[shadow_test_before_sample_index]
      shadow_test=shadow_test_before_sample.sample(n=int(num_of_class*train_num/2),replace=False)
      shadow_test_index=[item for item in shadow_test.index]

    #compute the Shannon entropy
    target_train_s_value=compute_shannon_entropy(num_of_class,target_train)
    target_test_s_value=compute_shannon_entropy(num_of_class,target_test)
    shadow_train_s_value=compute_shannon_entropy(num_of_class,shadow_train)
    shadow_test_s_value=compute_shannon_entropy(num_of_class,shadow_test)
 
    print("target_train[0].value_counts()")
    print(target_train[0].value_counts())
    print("target_test[0].value_counts()")
    print(target_test[0].value_counts())
    print("shadow_train[0].value_counts()")
    print(shadow_train[0].value_counts())
    print("shadow_test[0].value_counts()")
    print(shadow_test[0].value_counts())
    print("unique values:")
    print(target_train[0].nunique())
    print(target_test[0].nunique())
    print(shadow_train[0].nunique())
    print(shadow_test[0].nunique())
    target_train_m=[target_train[0].nunique(),len(target_train_index),target_train_s_value]
    target_test_m=[target_test[0].nunique(),len(target_test_index),target_test_s_value]
    shadow_train_m=[shadow_train[0].nunique(),len(shadow_train_index),shadow_train_s_value]
    shadow_test_m=[shadow_test[0].nunique(),len(shadow_test_index),shadow_test_s_value]

    #compute the metrics of datasets
    t_graph=to_networkx(total_dataset.data,to_undirected=True)
    t_dataset_m=compute_metric_of_graph(t_graph)
    target_dataset_s_value=compute_shannon_entropy(num_of_class,original_pd)
    target_dataset_m=[num_of_class,total_dataset.data.y.size()[0],target_dataset_s_value]+t_dataset_m

    return deepcopy(total_dataset),deepcopy(target_dataset_m),target_train_index,target_train_m,target_test_index,target_test_m,deepcopy(total_dataset),deepcopy(target_dataset_m),shadow_train_index,shadow_train_m,shadow_test_index,shadow_test_m #type 
  else:
    #from different datasets
    target_dataset=get_dataset(target_dataset_name,root_path)
    target_original_pd=pd.DataFrame(target_dataset.data.y.numpy())
    target_pd=pd.DataFrame(target_dataset.data.y.numpy())
    target_num_of_class=torch.max(target_dataset.data.y).item()+1
    shadow_dataset=get_dataset(shadow_dataset_name,shadow_dataset_name)
    shadow_original_pd=pd.DataFrame(shadow_dataset.data.y.numpy())
    shadow_pd=pd.DataFrame(shadow_dataset.data.y.numpy())
    shadow_num_of_class=torch.max(shadow_dataset.data.y).item()+1
    if sample_way==1:
      min_count=((target_pd[0].value_counts().min())//2)*2
      target_pd=target_pd.loc[[item[1] for item in target_pd.groupby(0).apply(pd.DataFrame.sample, n=min_count)[0].index]]
      target_index=[item for item in target_pd.index]
      target_train=target_pd.loc[[item[1] for item in target_pd.groupby(0).apply(pd.DataFrame.sample, frac=target_train_test_rate)[0].index]]
      target_train_index=[item for item in target_train.index]
      target_test_index=[item for item in target_index if item not in target_train_index]
      target_test=target_pd.loc[target_test_index]

      min_count=((shadow_pd[0].value_counts().min())//2)*2
      shadow_pd=shadow_pd.loc[[item[1] for item in shadow_pd.groupby(0).apply(pd.DataFrame.sample, n=min_count)[0].index]]
      shadow_index=[item for item in shadow_pd.index]
      shadow_train=shadow_pd.loc[[item[1] for item in shadow_pd.groupby(0).apply(pd.DataFrame.sample, frac=shadow_train_test_rate)[0].index]]
      shadow_train_index=[item for item in shadow_train.index]
      shadow_test_index=[item for item in shadow_index if item not in shadow_train_index]
      shadow_test=shadow_pd.loc[shadow_test_index]
    elif sample_way==0:
      target_index=[item for item in target_pd.index]
      target_train=target_pd.sample(frac=target_train_test_rate,replace = False)
      target_train_index=[item for item in target_train.index]
      target_test_index=[item for item in target_index if item not in target_train_index]
      target_test=target_pd.loc[target_test_index]

      shadow_index=[item for item in shadow_pd.index]
      shadow_train=shadow_pd.sample(frac=shadow_train_test_rate,replace = False)
      shadow_train_index=[item for item in shadow_train.index]
      shadow_test_index=[item for item in shadow_index if item not in shadow_train_index]
      shadow_test=shadow_pd.loc[shadow_test_index]
    elif sample_way==2:
      #number for training target and shadow models
      if target_dataset_name in len_of_each_class_in_train_data.keys():
        target_num=len_of_each_class_in_train_data[target_dataset_name]*2 #*2 is because the number of data points used for training target model doubles
      else:
        raise NotImplementedError
      target_index=[item for item in target_pd.index]
      target_train=target_pd.loc[[item[1] for item in target_pd.groupby(0).apply(pd.DataFrame.sample, n=target_num)[0].index]]
      target_train_index=[item for item in target_train.index]
      target_test_index_before_sample=[item for item in target_index if item not in target_train_index]
      target_test_before_sample=target_pd.loc[target_test_index_before_sample]
      target_test=target_test_before_sample.sample(n=target_num*target_num_of_class,replace = False)
      target_test_index=[item for item in target_test.index]

      if shadow_dataset_name in len_of_each_class_in_train_data.keys():
        shadow_num=len_of_each_class_in_train_data[shadow_dataset_name]*2
      else:
        raise NotImplementedError
      shadow_index=[item for item in shadow_pd.index]
      shadow_train=shadow_pd.loc[[item[1] for item in shadow_pd.groupby(0).apply(pd.DataFrame.sample, n=shadow_num)[0].index]]
      shadow_train_index=[item for item in shadow_train.index]
      shadow_test_index_before_sample=[item for item in shadow_index if item not in shadow_train_index]
      shadow_test_before_sample=shadow_pd.loc[shadow_test_index_before_sample]
      shadow_test=shadow_test_before_sample.sample(n=shadow_num*shadow_num_of_class,replace = False)
      shadow_test_index=[item for item in shadow_test.index]
    
    else:
      raise NotImplementedError

    target_train_s_value=compute_shannon_entropy(target_num_of_class,target_train)
    target_test_s_value=compute_shannon_entropy(target_num_of_class,target_test)
    shadow_train_s_value=compute_shannon_entropy(shadow_num_of_class,shadow_train)
    shadow_test_s_value=compute_shannon_entropy(shadow_num_of_class,shadow_test)
 
    print("target_train[0].value_counts()")
    print(target_train[0].value_counts())
    print("target_test[0].value_counts()")
    print(target_test[0].value_counts())
    print("shadow_train[0].value_counts()")
    print(shadow_train[0].value_counts())
    print("shadow_test[0].value_counts()")
    print(shadow_test[0].value_counts())

    target_train_m=[target_train[0].nunique(),len(target_train_index),target_train_s_value]
    target_test_m=[target_test[0].nunique(),len(target_test_index),target_test_s_value]
    shadow_train_m=[shadow_train[0].nunique(),len(shadow_train_index),shadow_train_s_value]
    shadow_test_m=[shadow_test[0].nunique(),len(shadow_test_index),shadow_test_s_value]

    #compute the metrics of datasets
    target_graph=to_networkx(target_dataset.data,to_undirected=True)
    t_dataset_m=compute_metric_of_graph(target_graph)
    target_dataset_s_value=compute_shannon_entropy(target_num_of_class,target_original_pd)
    target_dataset_m=[target_num_of_class,target_dataset.data.y.size()[0],target_dataset_s_value]+t_dataset_m

    shadow_graph=to_networkx(shadow_dataset.data,to_undirected=True)
    s_dataset_m=compute_metric_of_graph(shadow_graph)
    shadow_dataset_s_value=compute_shannon_entropy(shadow_num_of_class,shadow_original_pd)
    shadow_dataset_m=[shadow_num_of_class,shadow_dataset.data.y.size()[0],shadow_dataset_s_value]+s_dataset_m

    return target_dataset,target_dataset_m,target_train_index,target_train_m,target_test_index,target_test_m,shadow_dataset,shadow_dataset_m,shadow_train_index,shadow_train_m,shadow_test_index,shadow_test_m


#excute one MIA attack on specific datasets and models
def one_attack(target_parameters,shadow_parameters,root_path,target_shadow_rate,device,repeat_time,dir_name,w_explain,attack_type,sample_way,model_selection,fe_list_p): 
  """
  target_parmeters (dict): parameters for training target model
  shadow_parameters (dict): parameters for training shadow model
  w_explain (bool): whether explain the attack model
  sample_way (int): how to sample data points from dataset
  fe_list_p (int list): the features selection for attacking
  """
  repeat_result=[]
  repeat_result_p_0=[] #just hop_0
  repeat_result_p_1=[] #just hop_1
  repeat_result_p_2=[] #just hop_2
  repeat_result_p_3=[] #hop_0 and hop_1
  repeat_result_p_4=[] #hop_0 and hop_2
  repeat_result_p_5=[] #all probability prediction
  repeat_result_p_6=[] #res_all_cor,
  repeat_result_p_7=[] #res_p_o_g_t,
  repeat_result_p_8=[] #res_cro_entropy,
  repeat_result_p_9=[] #res_modified_cro_entropy

  target_acc=[] #[[test_acc,train_acc,gap]]
  shadow_acc=[]
  relabel_acc=[]
  
  target_train_metrics=[]
  target_test_metrics=[]
  shadow_train_metrics=[]
  shadow_test_metrics=[]

  target_dataset_metrics=[]
  shadow_dataset_metrics=[]

  attack_model_list=[]
  target_feature_list=[]
  shadow_feature_list=[]

  #attack_model_p_list=[]
  target_feature_p_list=[]
  shadow_feature_p_list=[]

  max_model=None
  max_ev_data=None
  max_sha_data=None
  max_attack_acc=0.0
  for r_t in range(0,repeat_time):
    #used for label-only attacking
    target_label_only_f=[]
    shadow_label_only_f=[]
    #used for prediction vector attacking
    target_pre_vec_f=[]
    shadow_pre_vec_f=[]
    #prepare dataset for target and shadow model
    target_dataset,target_dataset_metrics,target_train_index,target_train_m,target_test_index,target_test_m,shadow_dataset,shadow_dataset_metrics,shadow_train_index,shadow_train_m,shadow_test_index,shadow_test_m=prepare_dataset(target_parameters['dataset_name'],target_parameters['train_test_rate'],shadow_parameters['dataset_name'],shadow_parameters['train_test_rate'],root_path,target_shadow_rate,sample_way)
    
    #check the distribution of train and test index
    print("target_train-target_test:%s"%(str(len(set(target_train_index)-set(target_test_index)))))
    print("target_train-shadow_train:%s"%(str(len(set(target_train_index)-set(shadow_train_index)))))
    print("target_train-shadow_test:%s"%(str(len(set(target_train_index)-set(shadow_test_index)))))
    print("target_test-target_train:%s"%(str(len(set(target_test_index)-set(target_train_index)))))
    print("target_test-shadow_train:%s"%(str(len(set(target_test_index)-set(shadow_train_index)))))
    print("target_test-shadow_test:%s"%(str(len(set(target_test_index)-set(shadow_test_index)))))
    print("shadow_train-shadow_test:%s"%(str(len(set(shadow_train_index)-set(shadow_test_index)))))
    print("shadow_test-shadow_train:%s"%(str(len(set(shadow_test_index)-set(shadow_train_index)))))

    #remove number of neighbors and label in original task
    
    #train target model
    find_attack_features=True
    if attack_type==0:
      find_attack_features=False

    #train shadow model
    shadow_model,shadow_label_only_f,shadow_pre_of_all,shadow_test_acc,shadow_train_acc,shadow_gap,shadow_test_subgraph_metrics,shadow_train_subgraph_metrics=node_train(shadow_dataset,shadow_train_index,shadow_test_index,shadow_parameters,device,find_attack_features) #,shadow_ad_features
    shadow_acc.append([shadow_test_acc,shadow_train_acc,shadow_gap])

    
    #add the features of subgraphs
    #[num_of_class,num_of_nodes,shannon_value,is_connected,number_connected_components,degree_centrality_avg,degree_centrality_max,degree_centrality_min]
    shadow_train_m=shadow_train_m+shadow_train_subgraph_metrics
    shadow_test_m=shadow_test_m+shadow_test_subgraph_metrics
    shadow_train_metrics.append(shadow_train_m)
    shadow_test_metrics.append(shadow_test_m)
    
    target_model,target_label_only_f,target_pre_of_all,target_test_acc,target_train_acc,target_gap,target_test_subgraph_metrics,target_train_subgraph_metrics=node_train(target_dataset,target_train_index,target_test_index,target_parameters,device,find_attack_features) #,target_ad_features
    target_acc.append([target_test_acc,target_train_acc,target_gap])


    #add the features of subgraphs
    target_train_m=target_train_m+target_train_subgraph_metrics
    target_test_m=target_test_m+target_test_subgraph_metrics
    target_train_metrics.append(target_train_m)
    target_test_metrics.append(target_test_m)
    
    pro_feature_list=[]
    #if train_dataset_name==test_dataset_name relabel shadow_dataset with target model
    if target_parameters['dataset_name']==shadow_parameters['dataset_name']:
      
      #prediction vector method
      if attack_type==0 or attack_type==2:
        #implement prediction vector method
        # target_pre_vec_f[]
        for i_len in range(0,len(target_pre_of_all[target_train_index[0]]['p_w_a_t_o_t'])):
          pro_feature_list.append('p_'+str(i_len))
        #add keys of metrics for threshold-based MIAs 
        metric_key=['all_cor','p_o_g_t','cro_entropy','modified_cro_entropy']

        total_index=target_train_index+target_test_index+shadow_train_index+shadow_test_index
        for i_index in total_index:
          t_dic={}
          if i_index in target_train_index or i_index in shadow_train_index:
            t_dic['label']=1
          else:
            t_dic['label']=0
        
          for key in ['hop_0_1','hop_0_2','hop_1_1','hop_1_2','hop_2_1','hop_2_2']:
            if i_index in target_train_index or i_index in target_test_index:
              t_dic[key]=target_pre_of_all[i_index][key]
            else:
              t_dic[key]=shadow_pre_of_all[i_index][key]
          #deal with p_w_a_t_o_t
          for i_key in range(0,len(pro_feature_list)):
            if i_index in target_train_index or i_index in target_test_index:
              t_dic[pro_feature_list[i_key]]=target_pre_of_all[i_index]['p_w_a_t_o_t'][i_key]
            else:
              t_dic[pro_feature_list[i_key]]=shadow_pre_of_all[i_index]['p_w_a_t_o_t'][i_key]
          
          #deal with metrics
          for key_v in metric_key:
            if i_index in target_train_index or i_index in target_test_index:
              # print("target_pre_of_all[i_index].keys()")
              # print(target_pre_of_all[i_index].keys())
              t_dic[key_v]=target_pre_of_all[i_index][key_v]
            else:
              # print("shadow_pre_of_all[i_index].keys()")
              # print(shadow_pre_of_all[i_index].keys())
              t_dic[key_v]=shadow_pre_of_all[i_index][key_v]
          
          if i_index in target_train_index or i_index in target_test_index:
            target_pre_vec_f.append(t_dic)
          else:
            shadow_pre_vec_f.append(t_dic)
        
      if (attack_type==1 or attack_type==2) and (fe_list_p[0]==1 or fe_list_p[1]==1):
        #implement label-only method
        #relabel shadow dataset with target model
        relabelled_shadow_dataset=deepcopy(target_dataset)
        shadow_train_edge_index,_=subgraph(shadow_train_index,target_dataset.data.edge_index,relabel_nodes=True)
        shadow_train_data=deepcopy(target_dataset.data)
        shadow_train_data.x=shadow_train_data.x[shadow_train_index]
        shadow_train_data.y=shadow_train_data.y[shadow_train_index]
        shadow_train_data.edge_index=shadow_train_edge_index

        shadow_test_edge_index,_=subgraph(shadow_test_index,target_dataset.data.edge_index,relabel_nodes=True)
        shadow_test_data=deepcopy(target_dataset.data)
        shadow_test_data.x=shadow_test_data.x[shadow_test_index]
        shadow_test_data.y=shadow_test_data.y[shadow_test_index]
        shadow_test_data.edge_index=shadow_test_edge_index

        target_model.eval()
        shadow_train_data.to(device)
        shadow_test_data.to(device)
        relabelled_shadow_dataset.data.y[shadow_train_index]=target_model(shadow_train_data).max(dim=1)[1].detach().cpu()
        relabelled_shadow_dataset.data.y[shadow_test_index]=target_model(shadow_test_data).max(dim=1)[1].detach().cpu()
        del shadow_train_data,shadow_test_data
        print("Relabelled result for shadow dataset with target model")
        print(torch.unique(relabelled_shadow_dataset.data.y[shadow_train_index]))
        print(torch.unique(relabelled_shadow_dataset.data.y[shadow_test_index]))
        t_shadow_dataset_data=deepcopy(shadow_dataset.data)
        t_shadow_dataset_data.to(device)
        print(torch.unique(target_model(t_shadow_dataset_data).max(dim=1)[1].detach().cpu()[target_train_index+target_test_index]))
        del t_shadow_dataset_data
        relabelled_shadow_model,_,relabel_pre_of_all,r_test_acc,r_train_acc,r_gap,_,_=node_train(relabelled_shadow_dataset,shadow_train_index,shadow_test_index,shadow_parameters,device,False) #,shadow_ad_features
        relabel_acc.append([r_test_acc,r_train_acc,r_gap])
        
        #get the prediction of target data under shadow model
        target_under_shadow=get_0_1_2_hop_prediction(target_dataset,shadow_model,device,target_train_index,target_test_index)
        #get the prediction of target data under relabelled shadow model
        target_under_r_shadow=get_0_1_2_hop_prediction(target_dataset,relabelled_shadow_model,device,target_train_index,target_test_index)
        #use target model to get 1-hop, 0-hop, and 2-hop prediction vector for previous methods

        #target_features shadow_features
        for i in range(0,len(target_label_only_f)):
          # #just use top 2 in 0-hop and 2-hop
          # #print("target_under_r_shadow[target_label_only_f[i]['o_index']]")
          # #print(target_under_r_shadow[target_label_only_f[i]['o_index']])
          # target_label_only_f[i]['r_hop_0_1']=target_under_r_shadow[target_label_only_f[i]['o_index']]['hop_0_1']
          # target_label_only_f[i]['r_hop_0_2']=target_under_r_shadow[target_label_only_f[i]['o_index']]['hop_0_2']
          # # target_label_only_f[i]['r_hop_2_1']=target_under_r_shadow[target_label_only_f[i]['o_index']]['hop_2_1']
          # # target_label_only_f[i]['r_hop_2_2']=target_under_r_shadow[target_label_only_f[i]['o_index']]['hop_2_2']

          # target_label_only_f[i]['s_hop_0_1']=target_under_shadow[target_label_only_f[i]['o_index']]['hop_0_1']
          # target_label_only_f[i]['s_hop_0_2']=target_under_shadow[target_label_only_f[i]['o_index']]['hop_0_2']
          # # target_label_only_f[i]['s_hop_2_1']=target_under_shadow[target_label_only_f[i]['o_index']]['hop_2_1']
          # # target_label_only_f[i]['s_hop_2_2']=target_under_shadow[target_label_only_f[i]['o_index']]['hop_2_2']

          

          #use all probability
          #relabel
          for j in range(0,len(target_under_r_shadow[target_label_only_f[i]['o_index']]['hop_1_all'])):
            #target_label_only_f[i]['r_'+str(j)+'_hop_1']=target_under_r_shadow[target_label_only_f[i]['o_index']]['hop_1_all'][j]
            target_label_only_f[i]['r_'+str(j)+'_hop_0']=target_under_r_shadow[target_label_only_f[i]['o_index']]['hop_0_all'][j]
            #target_label_only_f[i]['r_'+str(j)+'_hop_2']=target_under_r_shadow[target_label_only_f[i]['o_index']]['hop_2_all'][j]
          #shadow
          for j in range(0,len(target_under_shadow[target_label_only_f[i]['o_index']]['hop_1_all'])):
            #target_label_only_f[i]['s_'+str(j)+'_hop_1']=target_under_shadow[target_label_only_f[i]['o_index']]['hop_1_all'][j]
            target_label_only_f[i]['s_'+str(j)+'_hop_0']=target_under_shadow[target_label_only_f[i]['o_index']]['hop_0_all'][j]
            #target_label_only_f[i]['s_'+str(j)+'_hop_2']=target_under_shadow[target_label_only_f[i]['o_index']]['hop_2_all'][j]
        
        for i in range(0,len(shadow_label_only_f)):
          # #just part of probability in 0-hop and 1-hop
          # shadow_label_only_f[i]['r_hop_0_1']=relabel_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_0_1']
          # shadow_label_only_f[i]['r_hop_0_2']=relabel_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_0_2']
          # # shadow_label_only_f[i]['r_hop_2_1']=relabel_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_2_1']
          # # shadow_label_only_f[i]['r_hop_2_2']=relabel_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_2_2']

          # shadow_label_only_f[i]['s_hop_0_1']=shadow_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_0_1']
          # shadow_label_only_f[i]['s_hop_0_2']=shadow_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_0_2']
          # # shadow_label_only_f[i]['s_hop_2_1']=shadow_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_2_1']
          # # shadow_label_only_f[i]['s_hop_2_2']=shadow_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_2_2']

          #use all probability
          #relabel
          for j in range(0,len(relabel_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_1_all'])):
            #shadow_label_only_f[i]['r_'+str(j)+'_hop_1']=relabel_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_1_all'][j]
            shadow_label_only_f[i]['r_'+str(j)+'_hop_0']=relabel_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_0_all'][j]
            #shadow_label_only_f[i]['r_'+str(j)+'_hop_2']=relabel_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_2_all'][j]
          #shadow
          for j in range(0,len(shadow_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_1_all'])):
            #shadow_label_only_f[i]['s_'+str(j)+'_hop_1']=shadow_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_1_all'][j]
            shadow_label_only_f[i]['s_'+str(j)+'_hop_0']=shadow_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_0_all'][j]
            #shadow_label_only_f[i]['s_'+str(j)+'_hop_2']=shadow_pre_of_all[shadow_label_only_f[i]['o_index']]['hop_2_all'][j]
        pass
    if len(target_label_only_f)>0:  
      target_label_only_f=pd.DataFrame(target_label_only_f)
      shadow_label_only_f=pd.DataFrame(shadow_label_only_f)
      # print("target_label_only_f['change_p_1.0'].value_counts()")
      # print(target_label_only_f['change_p_1.0'].value_counts())
      all_c_names=[col for col in target_label_only_f.columns]

      #print("all_c_name:%s"%(str(all_c_names)))

      num_of_class=torch.max(target_dataset.data.y).item()+1

      label_only_f_list=['label']
      #fe_list_0=['s_hop_0_1','s_hop_0_2'] only use two probability values
      fe_list_0=[] #use all prob
      fe_list_1=[] #use all prob
      for i in range(0,num_of_class):
        fe_list_0.append('s_'+str(i)+'_hop_0')
        fe_list_1.append('r_'+str(i)+'_hop_0')
      if fe_list_p[0]==1:
        label_only_f_list=label_only_f_list+fe_list_0
      #fe_list_1=['r_hop_0_1','r_hop_0_2'] only use two probability values
      if fe_list_p[1]==1:
        label_only_f_list=label_only_f_list+fe_list_1
      fe_list_2=['n_num','o_label','w_i_node']
      if fe_list_p[2]==1:
        label_only_f_list=label_only_f_list+fe_list_2
      #1.0,0.8,0.6,0.4,0.2,0.0
      fe_list_3=['i_none_max_0.0','i_none_max_0.2','i_none_max_0.4','i_none_max_0.6','i_none_max_0.8','i_none_max_1.0','i_none_min_0.0','i_none_min_0.2','i_none_min_0.4','i_none_min_0.6','i_none_min_0.8','i_none_min_1.0']
      if fe_list_p[3]==1:
        label_only_f_list=label_only_f_list+fe_list_3
      pre_list=['i_all_','i_step_','n_acc_all_','n_acc_none_','n_acc_avg_']
      mask_rates=[1.0,0.8,0.6,0.4,0.2,0.0]
      mask_sign=['min_','max_']
      fe_list_4=[]
      for pre_list_item in pre_list:
        for mask_sign_value in mask_sign:
          for mask_rates_item in mask_rates:
            fe_list_4.append(pre_list_item+mask_sign_value+str(mask_rates_item))
      if fe_list_p[4]==1:
        label_only_f_list=label_only_f_list+fe_list_4
      
      print("======================label-only start========================")
      # m_label_only_acc,m_label_only_pre,m_label_only_rec=0,0,0
      # m_label_only_model=None
      #dict[key] key means the model_selection
      attack_model,select_ev_acc,select_ev_pre,select_ev_rec,select_ev_auc,select_ev_f1,select_ev_low_fpr_tpr=train_and_ev_attack_model(shadow_label_only_f[label_only_f_list],target_label_only_f[label_only_f_list],model_selection)
      
      print("======================label-only end========================")
      repeat_result.append([select_ev_acc,select_ev_pre,select_ev_rec,select_ev_auc,select_ev_f1,select_ev_low_fpr_tpr])
      attack_model_list.append(attack_model)
      target_feature_list.append(target_label_only_f[label_only_f_list])
      shadow_feature_list.append(shadow_label_only_f[label_only_f_list])

      if model_selection!=5 and (max_model is None or max_attack_acc < select_ev_acc[model_selection]):
        max_model=attack_model[model_selection]
        max_ev_data=target_label_only_f[label_only_f_list]
        max_sha_data=shadow_label_only_f[label_only_f_list]
    
    
    if len(target_pre_vec_f)>0: 

      target_pre_vec_f=pd.DataFrame(target_pre_vec_f)
      shadow_pre_vec_f=pd.DataFrame(shadow_pre_vec_f)
      print("target_pre_vec_f['label'].value_counts()")
      print(target_pre_vec_f['label'].value_counts())
      print("shadow_pre_vec_f['label'].value_counts()")
      print(shadow_pre_vec_f['label'].value_counts())

      print("======================0-hop start========================")
      _,select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1=train_and_ev_attack_model(shadow_pre_vec_f[['hop_0_1','hop_0_2','label']],target_pre_vec_f[['hop_0_1','hop_0_2','label']],model_selection)
      
      print("======================0-hop end========================")
      repeat_result_p_0.append([select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1])
      
      print("======================2-hop start========================")
      _,select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1=train_and_ev_attack_model(shadow_pre_vec_f[['hop_2_1','hop_2_2','label']],target_pre_vec_f[['hop_2_1','hop_2_2','label']],model_selection)
      #print("select_ev_acc_p_2:%f, select_ev_pre_p_2:%f, select_ev_rec_p_2:%f"%(select_ev_acc_1,select_ev_pre_1,select_ev_rec_1))
      print("======================2-hop end========================")
      repeat_result_p_2.append([select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1])
      
      #0-hop with 2-hop combination
      print("======================0-hop 2-hop combination start========================")
      _,select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1=train_and_ev_attack_model(shadow_pre_vec_f[['hop_0_1','hop_0_2','hop_2_1','hop_2_2','label']],target_pre_vec_f[['hop_0_1','hop_0_2','hop_2_1','hop_2_2','label']],model_selection)
      print("======================0-hop 2-hop combination end========================")
      repeat_result_p_4.append([select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1])

      #all probability returned by the corresponding model
      print("======================all probability start========================")
      _,select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1=train_and_ev_attack_model(shadow_pre_vec_f[pro_feature_list+['label']],target_pre_vec_f[pro_feature_list+['label']],model_selection)
      repeat_result_p_5.append([select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1])
      print("======================all probability end========================")

      #metric-based MIAs
      metric_key=['all_cor','p_o_g_t','cro_entropy','modified_cro_entropy']
      print("======================metric-based start========================")
      res_all_cor,res_p_o_g_t,res_cro_entropy,res_modified_cro_entropy=select_threshold_and_evaluate(shadow_pre_vec_f[metric_key+['label']],target_pre_vec_f[metric_key+['label']])
      repeat_result_p_6.append(res_all_cor)
      repeat_result_p_7.append(res_p_o_g_t)
      repeat_result_p_8.append(res_cro_entropy)
      repeat_result_p_9.append(res_modified_cro_entropy)
      print("======================metric-based end========================")


      target_feature_p_list.append(target_pre_vec_f)
      shadow_feature_p_list.append(shadow_pre_vec_f)
    
  if len(attack_model_list)>0:
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    #save attack_model target_features
    for r_t in range(0,len(attack_model_list)):
      for selection_key in attack_model_list[r_t].keys():
        torch.save(attack_model_list[r_t][selection_key], dir_name+'attack_model_rp_'+str(r_t)+'_sk_'+str(selection_key)+'.pth')
      target_feature_list[r_t].to_csv(dir_name+'target_features_'+str(r_t)+'.csv')
      shadow_feature_list[r_t].to_csv(dir_name+'shadow_features_'+str(r_t)+'.csv')
  del attack_model_list,target_feature_list,shadow_feature_list

  if len(target_feature_p_list)>0:
    for r_t in range(0,len(target_feature_p_list)):
      #torch.save(attack_model_p_list[r_t], dir_name+'attack_model_p_'+str(r_t)+'.pth')
      target_feature_p_list[r_t].to_csv(dir_name+'target_features_p_'+str(r_t)+'.csv')
      shadow_feature_p_list[r_t].to_csv(dir_name+'shadow_features_p_'+str(r_t)+'.csv')

  #explain the model with hightest accuracy
  print("w_explain:%s"%(str(w_explain)))
  if model_selection != 5 and (max_model is not None) and w_explain:
    explain_attack_model(max_model,max_sha_data,max_ev_data,dir_name)
  del max_model,max_sha_data,max_ev_data
  return repeat_result,repeat_result_p_0,repeat_result_p_2,repeat_result_p_4,repeat_result_p_5,repeat_result_p_6,repeat_result_p_7,repeat_result_p_8,repeat_result_p_9,target_acc,shadow_acc,target_train_metrics,target_test_metrics,shadow_train_metrics,shadow_test_metrics,target_dataset_metrics,shadow_dataset_metrics

if __name__=='__main__':

  try:
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%d-%m-%H%M%S")
    target_parameters={}
    shadow_parameters={}
    cmd_args=vars(cmd_args)
    for key in cmd_args:
      if 'target_' in key:
        t_key=key[7:]
        target_parameters[t_key]=cmd_args[key]
      elif 'shadow' in key:
        t_key=key[7:]
        shadow_parameters[t_key]=cmd_args[key]
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))

    #check the whether has defense, if have defense, shadow model will also have same defense
    defend_sign=''
    if target_parameters['dropout']!=0.0 or target_parameters['norm'] is not None:
      defend_sign='_dropout_'+str(target_parameters['dropout'])+"_norm_"+str(target_parameters['norm'])
    dir_name='./results/node_res/'+target_parameters['dataset_name']+'_'+target_parameters['model_name']+'_'+shadow_parameters['dataset_name']+'_'+shadow_parameters['model_name']+'_'+date_time+defend_sign+'/'
    fe_list_p=[cmd_args['fe_list_0'],cmd_args['fe_list_1'],cmd_args['fe_list_2'],cmd_args['fe_list_3'],cmd_args['fe_list_4']]
    repeat_result,repeat_result_p_0,repeat_result_p_2,repeat_result_p_4,repeat_result_p_5,repeat_result_p_6,repeat_result_p_7,repeat_result_p_8,repeat_result_p_9,target_acc,shadow_acc,target_train_metrics,target_test_metrics,shadow_train_metrics,shadow_test_metrics,target_dataset_metrics,shadow_dataset_metrics=one_attack(target_parameters,shadow_parameters,cmd_args['root_path'],cmd_args['split_rate'],device,cmd_args['repeat_time'],dir_name,cmd_args['w_explain'],cmd_args['attack_type'],cmd_args['sample_way'],cmd_args['model_selection'],fe_list_p) #split_rate here means target_shadow_rate

    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    #constuct_file_name
    #train_acc 0, test_acc 1, train_loss 2, test_loss 3, the best 4, try all strategies 5
    num_to_select_name={
      0:'train_acc',
      1:'test_acc',
      2:'train_loss',
      3:'test_loss',
      4:'ev_acc',
    }
    file_name=dir_name+'attack_acc.txt'
    with open(file_name,'w')as f:
      f.write("********************Parameters For Current Experiment*********************\n")
      f.write(str(cmd_args)+"\n")
      f.write("********************End Parameters*********************\n")
      #for selection_key in [0,1,2,3,4]
      if len(repeat_result)>0 or len(repeat_result_p_0)>0:
        #determine key_list
        key_list=[]
        if len(repeat_result)>0:
          key_list=[item for item in repeat_result[0][0].keys()]
        elif len(repeat_result_p_0)>0:
          key_list=[item for item in repeat_result_p_0[0][0].keys()]

        #save the result of different methods
        for selection_key in key_list:
          t_r_r=[]
          broken_count=0 #the number of model always predict 0 or 1
          if len(repeat_result)>0:
            for repeat_time in range(0,len(repeat_result)):
              _t_item=[item[selection_key] for item in repeat_result[repeat_time]]
              if _t_item[1]==0.0 or _t_item[1]==1.0 or _t_item[2]==0.0 or _t_item[2]==1.0:
                broken_count=broken_count+1
              t_r_r.append(_t_item)
          t_r_r_p_0=[]
          t_r_r_p_2=[]
          t_r_r_p_4=[]
          t_r_r_p_5=[]
          broken_count_0=0
          broken_count_2=0
          broken_count_4=0
          broken_count_5=0
          if len(repeat_result_p_0)>0:
            for repeat_time in range(0,len(repeat_result_p_0)):
              _t_item_0=[item[selection_key] for item in repeat_result_p_0[repeat_time]]
              if _t_item_0[1]==0.0 or _t_item_0[1]==1.0 or _t_item_0[2]==0.0 or _t_item_0[2]==1.0:
                broken_count_0=broken_count_0+1
              t_r_r_p_0.append(_t_item_0)

              _t_item_2=[item[selection_key] for item in repeat_result_p_2[repeat_time]]
              if _t_item_2[1]==0.0 or _t_item_2[1]==1.0 or _t_item_2[2]==0.0 or _t_item_2[2]==1.0:
                broken_count_2=broken_count_2+1
              t_r_r_p_2.append(_t_item_2)

              _t_item_4=[item[selection_key] for item in repeat_result_p_4[repeat_time]]
              if _t_item_4[1]==0.0 or _t_item_4[1]==1.0 or _t_item_4[2]==0.0 or _t_item_4[2]==1.0:
                broken_count_4=broken_count_4+1
              t_r_r_p_4.append(_t_item_4)

              _t_item_5=[item[selection_key] for item in repeat_result_p_5[repeat_time]]
              if _t_item_5[1]==0.0 or _t_item_5[1]==1.0 or _t_item_5[2]==0.0 or _t_item_5[2]==1.0:
                broken_count_5=broken_count_5+1
              t_r_r_p_5.append(_t_item_5)


          f.write("\n##################model_selection:%s#######################"%(num_to_select_name[selection_key]))
          f.write("Label-only:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...] 0.5 as threshold for acc,pre,rec,f1\n")
          f.write(str(t_r_r)+'\n')
          f.write("Broken_count:"+str(broken_count)+"\n")
          if len(t_r_r)>0:
            f.write("Mean:"+str([round(item,3) for item in np.mean(t_r_r,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(t_r_r,axis=0).tolist()])+"\n")
          f.write("*****************************************\n")

          f.write("0-hop:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...] 0.5 as threshold for acc,pre,rec,f1\n")
          f.write(str(t_r_r_p_0)+'\n')
          f.write("Broken_count_0:"+str(broken_count_0)+"\n")
          if len(t_r_r_p_0)>0:
            f.write("Mean:"+str([round(item,3) for item in np.mean(t_r_r_p_0,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(t_r_r_p_0,axis=0).tolist()])+"\n")
          f.write("*****************************************\n")

          f.write("2-hop:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...] 0.5 as threshold for acc,pre,rec,f1\n")
          f.write(str(t_r_r_p_2)+'\n')
          f.write("Broken_count_2:"+str(broken_count_2)+"\n")
          if len(t_r_r_p_2)>0:
            f.write("Mean:"+str([round(item,3) for item in np.mean(t_r_r_p_2,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(t_r_r_p_2,axis=0).tolist()])+"\n")
          f.write("*****************************************\n")

          f.write("0-hop and 2-hop combination:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...] 0.5 as threshold for acc,pre,rec,f1\n")
          f.write(str(t_r_r_p_4)+'\n')
          f.write("Broken_count_4:"+str(broken_count_4)+"\n")
          if len(t_r_r_p_4)>0:
            f.write("Mean:"+str([round(item,3) for item in np.mean(t_r_r_p_4,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(t_r_r_p_4,axis=0).tolist()])+"\n")
          f.write("*****************************************\n")

          f.write("Use all probability values:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...] 0.5 as threshold for acc,pre,rec,f1\n")
          f.write(str(t_r_r_p_5)+'\n')
          f.write("Broken_count_5:"+str(broken_count_5)+"\n")
          if len(t_r_r_p_5)>0:
            f.write("Mean:"+str([round(item,3) for item in np.mean(t_r_r_p_5,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(t_r_r_p_5,axis=0).tolist()])+"\n")
          f.write("*****************************************\n")
    
    #save the result of metric-based MIA
    with open(file_name,'a')as f:
      #repeat_result_p_6,repeat_result_p_7,repeat_result_p_8,repeat_result_p_9
      #res_dict['all_cor'],res_dict['p_o_g_t'],res_dict['cro_entropy'],res_dict['modified_cro_entropy']
      f.write("\n##################metric-based MIA#######################")
      f.write("Prediction Correctness (predict with all training or testing nodes) for membership:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...]\n")
      f.write(str(repeat_result_p_6)+'\n')
      # f.write("Broken_count_5:"+str(broken_count_5)+"\n")
      if len(repeat_result_p_6)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(repeat_result_p_6,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(repeat_result_p_6,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

      f.write("Probability of ground-truth for membership:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...]\n")
      f.write(str(repeat_result_p_7)+'\n')
      # f.write("Broken_count_5:"+str(broken_count_5)+"\n")
      if len(repeat_result_p_7)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(repeat_result_p_7,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(repeat_result_p_7,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

      f.write("Cross Entropy for membership:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...]\n")
      f.write(str(repeat_result_p_8)+'\n')
      # f.write("Broken_count_5:"+str(broken_count_5)+"\n")
      if len(repeat_result_p_8)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(repeat_result_p_8,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(repeat_result_p_8,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

      f.write("Modified Cross Entropy for membership:[[accuracy,precision,recall,auc,f1,low_fpr_tpr]...]\n")
      f.write(str(repeat_result_p_9)+'\n')
      # f.write("Broken_count_5:"+str(broken_count_5)+"\n")
      if len(repeat_result_p_9)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(repeat_result_p_9,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(repeat_result_p_9,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

    acc_file_name=dir_name+'target_shadow_relabel_acc.txt'
    with open(acc_file_name,'w')as f:
      f.write("Target model performance:[[test_acc,train_acc,gap]...]\n")
      f.write(str(target_acc)+'\n')
      if len(target_acc)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(target_acc,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(target_acc,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

      #target_dataset_metrics,shadow_dataset_metrics
      f.write("Target dataset metrics:[[num_of_class,num_of_nodes,shannon_value,is_connected,number_connected_components,degree_centrality_avg,degree_centrality_max,degree_centrality_min]...]\n")
      f.write(str(target_dataset_metrics)+'\n')
      f.write("*****************************************\n")

      f.write("Target train metrics:[[num_of_class,num_of_nodes,shannon_value,is_connected,number_connected_components,degree_centrality_avg,degree_centrality_max,degree_centrality_min]...]\n")
      f.write(str(target_train_metrics)+'\n')
      if len(target_train_metrics)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(target_train_metrics,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(target_train_metrics,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

      f.write("Target test metrics:[[num_of_class,num_of_nodes,shannon_value,is_connected,number_connected_components,degree_centrality_avg,degree_centrality_max,degree_centrality_min]...]\n")
      f.write(str(target_test_metrics)+'\n')
      if len(target_test_metrics)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(target_test_metrics,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(target_test_metrics,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

      f.write("\n")

      f.write("Shadow model performance:[[test_acc,train_acc,gap]...]\n")
      f.write(str(shadow_acc)+'\n')
      if len(shadow_acc)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(shadow_acc,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(shadow_acc,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

      f.write("Shadow dataset metrics:[[num_of_class,num_of_nodes,shannon_value,is_connected,number_connected_components,degree_centrality_avg,degree_centrality_max,degree_centrality_min]...]\n")
      f.write(str(shadow_dataset_metrics)+'\n')
      f.write("*****************************************\n")

      f.write("Shadow train metrics:[[num_of_class,num_of_nodes,shannon_value,is_connected,number_connected_components,degree_centrality_avg,degree_centrality_max,degree_centrality_min]...]\n")
      f.write(str(shadow_train_metrics)+'\n')
      if len(shadow_train_metrics)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(shadow_train_metrics,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(shadow_train_metrics,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

      f.write("Target train metrics:[[num_of_class,num_of_nodes,shannon_value,is_connected,number_connected_components,degree_centrality_avg,degree_centrality_max,degree_centrality_min]...]\n")
      f.write(str(shadow_test_metrics)+'\n')
      if len(shadow_test_metrics)>0:
        f.write("Mean:"+str([round(item,3) for item in np.mean(shadow_test_metrics,axis=0).tolist()])+"---"+"Std:"+str([round(item,3) for item in np.std(shadow_test_metrics,axis=0).tolist()])+"\n")
      f.write("*****************************************\n")

  except Exception as err:
    print("Error occurs:")
    traceback.print_exc()



