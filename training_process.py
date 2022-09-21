from basic_gnn import GCN,GraphSAGE,GIN,GAT


from torch_geometric.nn.norm import BatchNorm,InstanceNorm,LayerNorm,GraphNorm,GraphSizeNorm,PairNorm,MessageNorm,DiffGroupNorm
from torch_geometric.loader import DataLoader
import torch
from time import time
from torch_geometric.utils import subgraph,to_networkx,from_networkx
from copy import deepcopy
import networkx as nx
from torch_geometric.data import Batch
import numpy as np
import random
from tqdm import tqdm
import sys
# import ctypes


#node classification task training
def node_train(t_dataset,train_index,test_index,model_ps,device,find_feature):
  """
  t_dataset (torch_geometric.datasets.XXdataset): Dataset used for training and testing model
  train_index (list): index list of training dataset
  test_index (list): index list of testing dataset
  model_ps (dict): parameters for the GNN model, keys include dataset_name, train_test_rate, model_name, optimizer_name
               criterion, batch_size, epoch, hidden_channels, num_layers, dropout, act, norm,
               jk, act_first, act_kwargs, readout;
  find_feature (bool): whether find the features of the dataset, True: find

  return:
  prediction_of_all (dict): the prediction (0-hp, 1-hop(all), and 2-hop) of model to all data points
  """
  dataset=deepcopy(t_dataset)
  #parse paramters
  in_channels=dataset.num_node_features
  out_channels=dataset.num_classes
  print("in_channels:%d, out_channels:%d"%(in_channels,out_channels))
  if model_ps['norm'] is None:
    norm=None
  elif model_ps['norm']=='BatchNorm':
    norm=BatchNorm(model_ps['hidden_channels'])
  elif model_ps['norm']=='LayerNorm':
    norm=LayerNorm(model_ps['hidden_channels'])
  else:
    raise NotImplementedError

  if model_ps['model_name']=='gcn':
    model=GCN(in_channels=in_channels,hidden_channels=model_ps['hidden_channels'],num_layers=model_ps['num_layers'],out_channels=out_channels,dropout=model_ps['dropout'],act=model_ps['act'],norm=norm,jk=model_ps['jk'],act_first=model_ps['act_first'],act_kwargs=model_ps['act_kwargs'],readout=model_ps['readout'])
  elif model_ps['model_name']=='graphsage':
    model=GraphSAGE(in_channels=in_channels,hidden_channels=model_ps['hidden_channels'],num_layers=model_ps['num_layers'],out_channels=out_channels,dropout=model_ps['dropout'],act=model_ps['act'],norm=norm,jk=model_ps['jk'],act_first=model_ps['act_first'],act_kwargs=model_ps['act_kwargs'],readout=model_ps['readout'])
  elif model_ps['model_name']=='gin':
    model=GIN(in_channels=in_channels,hidden_channels=model_ps['hidden_channels'],num_layers=model_ps['num_layers'],out_channels=out_channels,dropout=model_ps['dropout'],act=model_ps['act'],norm=norm,jk=model_ps['jk'],act_first=model_ps['act_first'],act_kwargs=model_ps['act_kwargs'],readout=model_ps['readout'])
  elif model_ps['model_name']=='gat':
    model=GAT(in_channels=in_channels,hidden_channels=model_ps['hidden_channels'],num_layers=model_ps['num_layers'],out_channels=out_channels,dropout=model_ps['dropout'],act=model_ps['act'],norm=norm,jk=model_ps['jk'],act_first=model_ps['act_first'],act_kwargs=model_ps['act_kwargs'],readout=model_ps['readout'])
  else:
    raise NotImplementedError
  model=model.to(device)
  print("model")
  print(model)
  # torch.nn.init.xavier_normal_(model.lin_l.weight, gain=1.0)
  #prepare of training
  if model_ps['optimizer_name']=='Adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=model_ps['learning_rate'],weight_decay=model_ps['mom_or_wd'])
  elif model_ps['optimizer_name']=='SGD':
    optimizer = torch.optim.SGD(model.parameters(),lr=model_ps['learning_rate'],momentum=model_ps['mom_or_wd'])
  else:
    raise NotImplementedError
  
  if model_ps['criterion']=='CrossEntropyLoss':
    criterion = torch.nn.CrossEntropyLoss()
  elif model_ps['criterion']=='NLLLoss':
    criterion = torch.nn.NLLLoss()
  else:
    raise NotImplementedError
  
  #divide the graph into two subgraphs for training and testing (only contains nodes in training or testing dataset)
  test_edge_index,_=subgraph(test_index,dataset.data.edge_index,relabel_nodes=True)
  test_data=deepcopy(dataset.data)
  test_data.x=test_data.x[test_index]
  test_data.y=test_data.y[test_index]
  test_data.edge_index=test_edge_index

  train_edge_index,_=subgraph(train_index,dataset.data.edge_index,relabel_nodes=True)
  train_data=deepcopy(dataset.data)
  train_data.x=train_data.x[train_index]
  train_data.y=train_data.y[train_index]
  train_data.edge_index=train_edge_index
  # print("train_edge_index:")
  # print(train_edge_index)

  graph_test=to_networkx(deepcopy(test_data),to_undirected=True)
  graph_train=to_networkx(deepcopy(train_data),to_undirected=True)

  #check whether the graph is the correct one
  # t_dict_of_list=nx.to_dict_of_lists(graph_train)
  # print("t_dict_of_list:")
  # print(t_dict_of_list)
  # t_adj=dataset.data.edge_index.numpy()
  # print("t_adj:")
  # print(t_adj)
  # for i in range(0,len(train_index)):
  #   t_list=[]
  #   for j in range(0,np.shape(t_adj)[1]):
  #     if (t_adj[0,j]==train_index[i] or t_adj[1,j]==train_index[i]) and t_adj[0,j] in train_index and t_adj[1,j] in train_index:
  #       t_list.append((t_adj[0,j],t_adj[1,j]))
  #   print("For index:%d"%(train_index[i]))
  #   print("Original:%s"%(str(t_list)))
  #   print("Original len:%d"%(len(t_list)))
  #   print("Change to subgraph:%s"%(str(t_dict_of_list[i])))
  #   print("Change len:%d"%(len(t_dict_of_list[i])))

  #   if len(t_list)!=(len(t_dict_of_list[i])*2):
  #     print("Error occur after changing")

  train_subgraph_metrics=compute_metric_of_graph(graph_train)
  test_subgraph_metrics=compute_metric_of_graph(graph_test)
  
  train_data_t=train_data.to(device)
  test_data_t=test_data.to(device)

  #training and testing
  best_loss, best_test_acc, best_train_acc, best_epoch, best_gap = 0, 0, 0, 0, 10
  best_model=None
  for epoch in range(0,model_ps['epoch']):
    model.train()
    optimizer.zero_grad()
    output = model(train_data_t)
    #output = torch.exp(model(train_data_t))
    loss = criterion(output,train_data_t.y)
    loss.backward()
    optimizer.step()

    model.eval()
    pred = model(test_data_t).max(dim=1)[1]
    correct=pred.eq(test_data_t.y).sum().item()
    test_acc=correct/len(test_index)

    model.eval()
    pred = model(train_data_t).max(dim=1)[1]
    correct = pred.eq(train_data_t.y).sum().item()
    train_acc=correct/len(train_index)
    gap=abs(train_acc-test_acc)
    if (test_acc>=best_test_acc) and (gap<=best_gap) and (epoch>=int(0.3*model_ps['epoch'])):
        best_loss, best_test_acc, best_train_acc, best_epoch, best_gap = loss, test_acc, train_acc, epoch, gap
        best_model=deepcopy(model)
    print('Epoch:{:03d}, Loss:{:04f}, Train acc:{:04f}, Test acc:{:04f}'.format(epoch,loss,train_acc,test_acc))
  print('best loss:{:04f}, best test acc:{:04f}, best train acc:{:04f}, epoch:{:03d}'.format(best_loss,best_test_acc,best_train_acc,best_epoch))
  del test_data_t,train_data_t

  #get predicton of all with 0-hop, 1-hop, and 2-hop subgraph as input
  prediction_of_all=get_0_1_2_hop_prediction(dataset,best_model,device,train_index,test_index)
   
  #get the prediction of train_dataset and test_dataset with all training or testing data points fed
  train_correct_count=0
  test_correct_count=0

  min_p_float=sys.float_info.min #avoid the err while computing cross-entropy

  best_model.eval()
  train_pre=torch.exp(best_model(train_data.to(device)).detach().cpu())
  for i in range(0,len(train_index)):
    prediction_list=train_pre[i].numpy().tolist()
    prediction_label=torch.argmax(train_pre[i]).item()
    assert torch.max(train_pre[i]).item()==prediction_list[prediction_label]
    ground_truth=dataset.data.y[train_index[i]].item()

    prediction_of_all[train_index[i]]['p_w_a_t_o_t']=prediction_list
    #prediction_of_all[train_index[i]]['p_w_a_t_o_t']=train_pre[i].numpy().tolist()
    #obtain prediction correctness and probability of ground-truth; compute cross-entropy, modified cross-entropy;
    #key ['all_cor','p_o_g_t','cro_entropy','modified_cro_entropy']

    prediction_of_all[train_index[i]]['p_o_g_t']=prediction_list[ground_truth]
    cro_entropy=0.0
    modified_cro_entropy=0.0
    for j in range(0,len(prediction_list)):
      c_v=prediction_list[j]
      if prediction_list[j]<min_p_float:
        c_v=min_p_float
      t=(-c_v*np.log(c_v))
      cro_entropy=cro_entropy+t
      if j==ground_truth:
        modified_cro_entropy=modified_cro_entropy+(-(1-c_v)*np.log(c_v))
      else:
        modified_cro_entropy=modified_cro_entropy+t
    
    prediction_of_all[train_index[i]]['cro_entropy']=cro_entropy
    prediction_of_all[train_index[i]]['modified_cro_entropy']=modified_cro_entropy
    
    
    if prediction_label==ground_truth:
      train_correct_count=train_correct_count+1
      prediction_of_all[train_index[i]]['all_cor']=1
    else:
      prediction_of_all[train_index[i]]['all_cor']=0
  
  best_model.eval()
  test_pre=torch.exp(best_model(test_data.to(device)).detach().cpu())
  for i in range(0,len(test_index)):
    prediction_list=test_pre[i].numpy().tolist()
    prediction_label=torch.argmax(test_pre[i]).item()
    assert torch.max(test_pre[i]).item()==prediction_list[prediction_label]
    ground_truth=dataset.data.y[test_index[i]].item()

    prediction_of_all[test_index[i]]['p_w_a_t_o_t']=prediction_list
    #obtain probability of ground-truth; compute cross-entropy, modified cross-entropy;

    prediction_of_all[test_index[i]]['p_o_g_t']=prediction_list[ground_truth]
    cro_entropy=0.0
    modified_cro_entropy=0.0
    for j in range(0,len(prediction_list)):
      c_v=prediction_list[j]
      if prediction_list[j]<min_p_float:
        c_v=min_p_float
      t=(-c_v*np.log(c_v))
      cro_entropy=cro_entropy+t
      if j==ground_truth:
        modified_cro_entropy=modified_cro_entropy+(-(1-c_v)*np.log(c_v))
      else:
        modified_cro_entropy=modified_cro_entropy+t
    
    prediction_of_all[test_index[i]]['cro_entropy']=cro_entropy
    prediction_of_all[test_index[i]]['modified_cro_entropy']=modified_cro_entropy
    
    if prediction_label==ground_truth:
      test_correct_count=test_correct_count+1
      prediction_of_all[test_index[i]]['all_cor']=1
    else:
      prediction_of_all[test_index[i]]['all_cor']=0

  print("Prediction train_acc:%.3f test_acc:%.3f"%(train_correct_count/len(train_index),test_correct_count/len(test_index)))

  if find_feature is not True:
    return best_model,[],prediction_of_all,best_test_acc,best_train_acc,best_gap,test_subgraph_metrics,train_subgraph_metrics
  
  graph_n=to_networkx(dataset.data,to_undirected=True)
  adj_dict=nx.to_dict_of_lists(graph_n)

  feature_list=[]
  total_index=train_index+test_index
  # signal=0
  mask_rate=[1.0,0.8,0.6,0.4,0.2,0.0] #more rates of masking
  #mask_rate=[1.0,0.5,0.0]
  for i_th in tqdm(range(len(total_index)),desc='Obtaining the attack features'):
    index_num=total_index[i_th]
    t={}
    t['o_index']=index_num
    # print(type(dataset.data.y[index_num].item()))
    # print(dataset.data.y[index_num].item())

    #[FIXME] test without this feature
    t['o_label']=dataset.data.y[index_num].item()
    if index_num in train_index:
      t['label']=1
    elif index_num in test_index:
      t['label']=0
    #find the neighbors of this data point, the neighbors might be members or non-members, but not data points from target(training on shadow).
    #Can we relax the limitation that the neighbors just from target or shadow? I think not
    neighbors=[item for item in adj_dict[index_num] if item in total_index]
    if index_num in neighbors:
      t['w_s_loop']=1
      
      #[FIXME] test without this feature
      t['n_nums']=len(neighbors)-1
      
      if len(neighbors)==1: #just self
        t['w_i_node']=1
      else:
        t['w_i_node']=0
    else:
      t['w_s_loop']=0

      #[FIXME] test without this feature
      t['n_num']=len(neighbors)
      
      if len(neighbors)>0:
        t['w_i_node']=0
      else:
        t['w_i_node']=1
    #w_s_loop: check whether this node has self-loop, if none, add self-loop
    #n_nums: the 1-hop neighbors of this node
    #w_i_node: whether this node is isolated node

    if t['w_s_loop']==0:
      neighbors.append(index_num) #add self-loop
    #change the index_num to first one
    #print("neighbors:%s before moving"%(str(neighbors)))
    t_index=neighbors.index(index_num)
    for i in range(t_index,0,-1):
      neighbors[i]=neighbors[i-1]
    neighbors[0]=index_num
    #print("neighbors:%s after moving"%(str(neighbors)))

    # t_neis=[index_num]
    for rate in mask_rate: #apply different mask rate to the feature of node
      #print(dataset.data.x.size(1))
      feature_len=dataset.data.x.size(1)
      mask_list=random.sample([i for i in range(0,feature_len)],int(feature_len*rate))
      t_feature=deepcopy(dataset.data.x[index_num])
      index_feature=dataset.data.x[index_num]

      #[FIXME] change 0.0 to 1.0 because most of values are 0.0
      
      mask_value_list=[0.0,1.0]
      mask_sign=['min_','max_']

      for mask_i in range(0,len(mask_value_list)):
        mask_value=mask_value_list[mask_i]
        index_feature[mask_list]=torch.tensor([mask_value]*len(mask_list)) #mask some features of current data point
      
        #add the actual altered rate for masking
        change_count=0
        for c_i in range(0,feature_len):
          # if t_feature[c_i].item()!=0.0 or index_feature[c_i].item()!=0.0:
          #   print("******")
          #   print("t_feature[c_i].item()")
          #   print(t_feature[c_i].item())
          #   print("index_feature[c_i].item()")
          #   print(index_feature[c_i].item())
          #   print("******")
          if (t_feature[c_i].item())!=(index_feature[c_i].item()):
            change_count=change_count+1
        change_pro=change_count/feature_len
        # if change_pro>0:
        #   print("change_pro:%f"%(change_pro))
        t["change_p_"+str(rate)]=change_pro
        
        i_correct=[]
        n_acc=[]
        t_neis=[index_num]
        for n_i in neighbors:
          if n_i!=index_num:
            t_neis.append(n_i)
            t_dic={}
            t_dic[index_num]=t_neis
            g=nx.from_dict_of_lists(t_dic)
            pyg_data=from_networkx(g)
          else:
            t_=[index_num] #Just self-loop
            t_dic={}
            t_dic[index_num]=t_
            g=nx.from_dict_of_lists(t_dic)
            pyg_data=from_networkx(g)
          pyg_data.x=dataset.data.x[neighbors] #the first one is always be current data point
          pyg_data.x[0]=index_feature
          pyg_data.y=dataset.data.y[neighbors]
          #compute i_all i_none i_steps n_acc_all n_acc_none n_acc_avg
          #if isolated node, i_all=i_none=i_steps n_acc_all=n_acc_none=1
          #i_all: keep the connections between this node with its neighbors
          #i_none: remove all the edges related with this node
          #i_steps: remove the neighbor step by step, the accuracy of current node
          #n_acc_all: the accuracy of neighbors while keeping all the edges
          #n_acc_none: the accuracy of neighbors while removing all the edges
          #n_acc_avg: the avg accuracy of neighbors while removing the edges step by step
          best_model.eval()
          # pyg_data.to(device)
          pred=best_model(pyg_data.to(device)).max(dim=1)[1]
          correct=pred.eq(pyg_data.y).detach()
          i_correct.append(1.0 if correct[0].item() else 0.0)
          if len(neighbors)==1:#isolated node
            n_acc.append(1.0)
          else:
            n_acc.append((correct[1:].sum()/(len(correct)-1)).item())
        t['i_none_'+mask_sign[mask_i]+str(rate)]=i_correct[0]
        t['i_all_'+mask_sign[mask_i]+str(rate)]=i_correct[len(neighbors)-1]
        t['i_step_'+mask_sign[mask_i]+str(rate)]=np.sum(i_correct)/(len(i_correct))
        t['n_acc_all_'+mask_sign[mask_i]+str(rate)]=n_acc[len(neighbors)-1]
        t['n_acc_none_'+mask_sign[mask_i]+str(rate)]=n_acc[0]
        t['n_acc_avg_'+mask_sign[mask_i]+str(rate)]=np.average(n_acc)
    feature_list.append(t)
  #print(feature_list[0])
  #return best_model,feature_list,prediction_of_all,train_pre,test_pre,best_test_acc,best_train_acc,best_gap
  return best_model,feature_list,prediction_of_all,best_test_acc,best_train_acc,best_gap,test_subgraph_metrics,train_subgraph_metrics


def get_0_1_2_hop_prediction(dataset,model,device,train_index,test_index):
  #get 0-hop 1-hop(all) 2-hop prediction of the graph
  #parameter dataset,model,device
  prediction_of_all={}
  graph_n=to_networkx(dataset.data,to_undirected=True)
  adj_dict=nx.to_dict_of_lists(graph_n)
  #index_all=[i for i in range(0,dataset.data.y.size(0))]
  index_all=train_index+test_index

  def get_prediction_of_all(index_t,t_dict,model,device):
    g=nx.from_dict_of_lists(t_dic)
    pyg_data=from_networkx(g)
    pyg_data.x=dataset.data.x[index_t] #the first one is always be current data point
    pyg_data.y=dataset.data.y[index_t]
    model.eval()
    pre_=torch.exp(model(pyg_data.to(device)).detach().cpu())
    #pre_=torch.nn.functional.softmax(model(pyg_data.to(device)).detach().cpu(),dim=1)

    #judge whether prediction is the same with ground-truth

    #obtain probability of ground-truth; compute cross-entropy, modified cross-entropy;

    return pre_[0] #current index is always on the first one
  
  #print(len(index_all))
  for i in tqdm(range(0,len(index_all)),desc='Get 0-hop, 1-hop, and 2-hop res'):
    save_dic={}
    
    index_num=index_all[i]
    save_dic['o_index']=index_num
    #0-hop
    index_t=[index_num]
    t_dic={}
    t_dic[index_num]=[index_num]#just self loop
    hop_0_res=get_prediction_of_all(index_t,t_dic,model,device).numpy()
    #print("hop_0_res:%s"%(str(hop_0_res)))
    hop_0=-np.partition(-hop_0_res,2)[:2]
    save_dic['hop_0_1']=hop_0[0]
    save_dic['hop_0_2']=hop_0[1]
    save_dic['hop_0_all']=hop_0_res.tolist()
    #print("hop_0_all:%s"%(str(save_dic['hop_0_all'])))
    
    #1-hop
    t_list=[nei for nei in adj_dict[index_num] if nei in index_all]#just use the data for training and testing current model
    #if no self loop, add
    if index_num not in t_list:
      t_list.append(index_num)
    #change the first one to current index
    t_index=t_list.index(index_num)
    for j in range(t_index,0,-1):
      t_list[j]=t_list[j-1]
    t_list[0]=index_num
    t_dic={}
    t_dic[index_num]=t_list
    hop_1_list=deepcopy(t_list)#for 2-hop
    hop_1_res=get_prediction_of_all(hop_1_list,t_dic,model,device).numpy()
    # print("hop_1_res:%s"%(str(hop_1_res)))
    hop_1=-np.partition(-hop_1_res,2)[:2]
    save_dic['hop_1_1']=hop_1[0]
    save_dic['hop_1_2']=hop_1[1]
    save_dic['hop_1_all']=hop_1_res.tolist()
    #print("hop_1_all:%s"%(str(save_dic['hop_1_all'])))

    #2-hop
    t_dic={}
    t_dic[index_num]=hop_1_list
    #t_list=deepcopy(hop_1_list) #the order of node changed after generate graph from dict
    for item in hop_1_list:
      if item != index_num:
        t_dic[item]=[nei for nei in adj_dict[item] if nei in index_all]
    #nx.from_dict_of_lists will change the order of node as the encount order, key and item in map[key]
    t_list=[]
    for key in t_dic:
      t_list.append(key)
    for key in t_dic:
      for n_i in t_dic[key]:
        if n_i not in t_list:
          t_list.append(n_i)
    hop_2_res=get_prediction_of_all(t_list,t_dic,model,device).numpy()
    # print("hop_2_res:%s"%(str(hop_2_res)))
    hop_2=-np.partition(-hop_2_res,2)[:2]
    save_dic['hop_2_1']=hop_2[0]
    save_dic['hop_2_2']=hop_2[1]
    save_dic['hop_2_all']=hop_2_res.tolist()
    # print("save_dic:%s"%(str(save_dic)))
    #print("hop_2_all:%s"%(str(save_dic['hop_2_all'])))

    prediction_of_all[index_num]=save_dic
  return prediction_of_all
    