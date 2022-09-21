import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,roc_auc_score,f1_score,roc_curve
from bisect import bisect_left
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer

class model_attack(nn.Module):
  def __init__(self, input_n, hidden_n, output_n):
    super(model_attack, self).__init__()
    ## define the layer                
    self.linear1 = torch.nn.Linear(input_n, hidden_n)
    self.linear2 = torch.nn.Linear(hidden_n, hidden_n)
    self.linear3 = torch.nn.Linear(hidden_n, output_n)
    #self.softmax_f=F.softmax(dim=1)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.1)
    self.batchnorm1 = nn.BatchNorm1d(hidden_n)
    self.batchnorm2 = nn.BatchNorm1d(hidden_n)

  def forward(self, x): 
    x=self.relu(self.linear1(x))
    x=self.batchnorm1(x)
    x=self.relu(self.linear2(x))
    x=self.batchnorm2(x)
    x=self.linear3(x)
    x=self.dropout(x)
    x=torch.nn.Sigmoid()(x)
    return x

def set_value_for_dic_list(key,test_list_t,test_value_t,train_list_t,train_value_t,ev_list_t,ev_value_t):
  
  test_list=deepcopy(test_list_t)
  test_value=deepcopy(test_value_t)
  train_list=deepcopy(train_list_t)
  train_value=deepcopy(train_value_t)
  ev_list=deepcopy(ev_list_t)
  ev_value=deepcopy(ev_value_t)

  for i in range(0,len(test_list)):
    test_list[i][key]=test_value[i]
  
  for i in range(0,len(train_list)):
    train_list[i][key]=train_value[i]
  
  for i in range(0,len(ev_list)):
    ev_list[i][key]=ev_value[i]

  return test_list, train_list, ev_list

def display(key,select_test_metrics,select_train_metrics,select_ev_metrics):
  test_show_value=[item[key] for item in select_test_metrics]
  train_show_value=[item[key] for item in select_train_metrics]
  ev_show_value=[item[key] for item in select_ev_metrics]
  print("select_test_loss:%.3f,select_test_acc:%.3f,select_test_pre:%.3f,select_test_rec:%.3f,select_test_f1:%.3f,select_test_TN:%d,select_test_FP:%d,select_test_FN:%d,select_test_TP:%d"%(test_show_value[0],test_show_value[1],test_show_value[2],test_show_value[3],test_show_value[4],test_show_value[5],test_show_value[6],test_show_value[7],test_show_value[8]))
  print("select_train_loss:%.3f,select_train_acc:%.3f,select_train_pre:%.3f,select_train_rec:%.3f,select_train_f1:%.3f,select_train_TN:%d,select_train_FP:%d,select_train_FN:%d,select_train_TP:%d"%(train_show_value[0],train_show_value[1],train_show_value[2],train_show_value[3],train_show_value[4],train_show_value[5],train_show_value[6],train_show_value[7],train_show_value[8]))
  print("select_ev_loss:%.3f,select_ev_acc:%.3f,select_ev_pre:%.3f,select_ev_rec:%.3f,select_ev_auc:%.3f,select_ev_f1:%.3f,select_ev_TN:%d,select_ev_FP:%d,select_ev_FN:%d,select_ev_TP:%d,select_ev_low_fpr_tpr:%.3f"%(ev_show_value[0],ev_show_value[1],ev_show_value[2],ev_show_value[3],ev_show_value[4],ev_show_value[5],ev_show_value[6],ev_show_value[7],ev_show_value[8],ev_show_value[9],ev_show_value[10]))
#training process of attack model
#define the function of training model
#train single model
def train_one_attack_dataset(train_data,test_data,ev_data,epoch_num,batch_size,model,x_list,y_list,device,optimizer,scheduler,criterion,model_selection): #return model 
  """
  train_data, DataFrame, the data used for training
  test_data, DataFrame, the data used for testing
  ev_data, DataFrame, the data used for evaluation eval_data
  epoch_num, int, the number of epoch of training
  batch_size, int, the number of data samples used for updating weights
  model, torch.nn.Model, the model for training
  x_list, list, the column name of feature
  y_list, list, column name of label
  device, torch.device, the device used for training model
  optimizer,torch.optim.XXX, the optimizer used for training
  criterion, torch.nn.XXX, the loss function
  model_selection, int, specify the method of selecting the attack model; selecting attack model with different metric (train_acc 0, test_acc 1, train_loss 2, test_loss 3, the best 4, try all strategies 5)
  """
  rows,colus=train_data.shape

  best_model={}
  select_test_loss={}
  select_test_acc={}
  select_test_pre={}
  select_test_rec={}
  select_test_f1={}
  select_test_TN={}
  select_test_FP={}
  select_test_FN={}
  select_test_TP={}
  select_test_metrics=[select_test_loss,select_test_acc,select_test_pre,select_test_rec,select_test_f1,select_test_TN,select_test_FP,select_test_FN,select_test_TP]

  select_train_loss={}
  select_train_acc={}
  select_train_pre={}
  select_train_rec={}
  select_test_f1={}
  select_train_TN={}
  select_train_FP={}
  select_train_FN={}
  select_train_TP={}
  select_train_metrics=[select_train_loss,select_train_acc,select_train_pre,select_train_rec,select_test_f1,select_train_TN,select_train_FP,select_train_FN,select_train_TP]
  
  select_ev_loss={}
  select_ev_acc={}
  select_ev_pre={}
  select_ev_rec={}
  select_ev_auc={}
  select_ev_f1={}
  select_ev_TN={}
  select_ev_FP={}
  select_ev_FN={}
  select_ev_TP={}
  select_ev_low_fpr_tpr={}
  select_ev_metrics=[select_ev_loss,select_ev_acc,select_ev_pre,select_ev_rec,select_ev_auc,select_ev_f1,select_ev_TN,select_ev_FP,select_ev_FN,select_ev_TP,select_ev_low_fpr_tpr]

  init_train_value=[100,0,0,0,0,0,0,0,0]
  init_test_value=[100,0,0,0,0,0,0,0,0]
  init_ev_value=[100,0,0,0,0,0,0,0,0,0,0]
  
  if model_selection!=5:
    select_test_metrics,select_train_metrics,select_ev_metrics=set_value_for_dic_list(model_selection,select_test_metrics,init_test_value,select_train_metrics,init_train_value,select_ev_metrics,init_ev_value)
    best_model[model_selection]=None
  if model_selection==5:
    for k_v in [0,1,2,3,4]:
      select_test_metrics,select_train_metrics,select_ev_metrics=set_value_for_dic_list(k_v,select_test_metrics,init_test_value,select_train_metrics,init_train_value,select_ev_metrics,init_ev_value)
      best_model[k_v]=None

  #print("********************start to train one model***********************")
  for epoch in range(epoch_num):  # loop over the dataset multiple times
    running_loss = 0.0
    left_point=0
    #left_point=rows-1 #just test one example
    train_data=train_data.sample(frac=1) #shuffle data in different epoch
    
    model.train()
    while left_point!=rows:
      right_point=min([rows,left_point+batch_size])
      batch_data=train_data.iloc[left_point:right_point,:]

      #the purpose of reshaping is change one demension to 2 demension
      data_size=right_point-left_point
      temp=batch_data[x_list].values
      feature=torch.from_numpy(temp).float().to(device)
      del temp

      label=torch.from_numpy(batch_data[y_list].values).float().to(device)
      del batch_data
      
      optimizer.zero_grad()
      outputs= model(feature)
      loss = criterion(outputs,label) #loss sum of a batch
      loss.backward()
      optimizer.step()
      #scheduler.step()
      del outputs
      del label
      
      left_point=right_point
      running_loss += float(loss)*data_size#loss sum of all data points

    model.eval()
    with torch.set_grad_enabled(False):
      #evaluate test data
      t_d_size=test_data.shape[0]
      t_1=test_data[x_list].values
      test_f=torch.from_numpy(t_1).float().to(device)
      del t_1
      #test_output=model(test_f)
      test_output= model(test_f)
      _predict=[1 if item>=0.5 else 0 for item in torch.squeeze(test_output).detach().cpu().numpy().tolist()]
      _true=torch.squeeze(torch.from_numpy(test_data[y_list].values)).detach().cpu().numpy().tolist()

      test_pro=torch.squeeze(test_output).detach().cpu().numpy().tolist()
      test_auc=roc_auc_score(_true,test_pro)
      test_f1=f1_score(_true,_predict)
      test_TN,test_FP,test_FN,test_TP=confusion_matrix(_true,_predict).ravel()

      test_acc=accuracy_score(_true,_predict)
      test_pre=precision_score(_true,_predict,zero_division=0)
      test_rec=recall_score(_true,_predict,zero_division=0)
      #print("***test accuracy:%.3f--test precision:%.3f--test recall:%.3f****"%(test_acc,test_pre,test_rec))
      del test_f
      test_label=torch.from_numpy(test_data[y_list].values).float().to(device)
      #test_acc=report_metric(test_output,test_label,metric)
      test_loss = criterion(test_output, test_label)
      
      #evaluate train data
      r_d_size=train_data.shape[0]
      t_2=train_data[x_list].values
      train_f=torch.from_numpy(t_2).float().to(device)
      del t_2
      train_output=model(train_f)
      _predict=[1 if item>=0.5 else 0 for item in torch.squeeze(train_output).detach().cpu().numpy().tolist()]
      _true=torch.squeeze(torch.from_numpy(train_data[y_list].values)).detach().cpu().numpy().tolist()
      train_pro=torch.squeeze(train_output).detach().cpu().numpy().tolist()
      train_auc=roc_auc_score(_true,train_pro)
      train_f1=f1_score(_true,_predict)
      train_TN,train_FP,train_FN,train_TP=confusion_matrix(_true,_predict).ravel()
      #print("train--TP:%d TN:%d FP:%d FN:%d AUC:%.3f F1:%.3f"%(TP,TN,FP,FN,train_auc,train_f1))
      train_acc=accuracy_score(_true,_predict)
      train_pre=precision_score(_true,_predict,zero_division=0)
      train_rec=recall_score(_true,_predict,zero_division=0)
      #print("***train accuracy:%.3f--train precision:%.3f--train recall:%.3f****"%(train_acc,train_pre,train_rec))
      del train_f
      train_label=torch.from_numpy(train_data[y_list].values).float().to(device)
      train_loss = criterion(train_output, train_label)

      #ev data performance after the final epoch
      r_d_size=ev_data.shape[0]
      t_3=ev_data[x_list].values
      ev_f=torch.from_numpy(t_3).float().to(device)
      del t_3
      ev_output=model(ev_f)
      _predict=[1 if item>=0.5 else 0 for item in torch.squeeze(ev_output).detach().cpu().numpy().tolist()]
      ev_prob=torch.squeeze(ev_output).detach().cpu().numpy().tolist()
      _true=torch.squeeze(torch.from_numpy(ev_data[y_list].values)).detach().cpu().numpy().tolist()

      ev_auc=roc_auc_score(_true,ev_prob,average='weighted') #if error, regard the ev_auc as 0.5

      ev_f1=f1_score(_true,_predict)
      ev_TN,ev_FP,ev_FN,ev_TP=confusion_matrix(_true,_predict).ravel()
      fpr, tpr, thresholds = roc_curve(_true, ev_prob)
      fp_list=[0.001,0.01,0.1]
      tp_list=[]
      for item in fp_list:
        fp_index=bisect_left(fpr,item)
        if fp_index==0:
          tp_list.append(tpr[0])
        elif fp_list==len(fpr):
          raise ValueError("Give fpr larger than 1.0")
        else:
          tp_list.append(tpr[fp_index-1])
      ev_low_fpr_tpr=tp_list[1] #if error occurs, regard the ev_low_fpr_tpr as fpr 0.01
      ev_acc=accuracy_score(_true,_predict)
      ev_pre=precision_score(_true,_predict,zero_division=0)
      ev_rec=recall_score(_true,_predict,zero_division=0)
      #print("***ev accuracy:%.3f--ev precision:%.3f--ev recall:%.3f****"%(ev_acc,ev_pre,ev_rec))
      del ev_f
      ev_label=torch.from_numpy(ev_data[y_list].values).float().to(device)
      ev_loss = criterion(ev_output, ev_label)
    
    current_train_value=[train_loss,train_acc,train_pre,train_rec,test_f1,train_TN,train_FP,train_FN,train_TP]
    current_test_value=[test_loss,test_acc,test_pre,test_rec,test_f1,test_TN,test_FP,test_FN,test_TP]
    current_ev_value=[ev_loss,ev_acc,ev_pre,ev_rec,ev_auc,ev_f1,ev_TN,ev_FP,ev_FN,ev_TP,ev_low_fpr_tpr]

    #(train_acc 0, test_acc 1, train_loss 2, test_loss 3, the best 4, try all strategies 5)
    if (model_selection==0 or model_selection==5) and train_acc>select_train_metrics[1][0]: #first index determines loss, acc; the second index determines model_selection
      best_model[0]=deepcopy(model)
      select_test_metrics,select_train_metrics,select_ev_metrics=set_value_for_dic_list(0,select_test_metrics,current_test_value,select_train_metrics,current_train_value,select_ev_metrics,current_ev_value)

    if (model_selection==1 or model_selection==5) and test_acc>select_test_metrics[1][1]:
      best_model[1]=deepcopy(model)
      select_test_metrics,select_train_metrics,select_ev_metrics=set_value_for_dic_list(1,select_test_metrics,current_test_value,select_train_metrics,current_train_value,select_ev_metrics,current_ev_value)

    if (model_selection==2 or model_selection==5) and train_loss<select_train_metrics[0][2]:
      best_model[2]=deepcopy(model)
      select_test_metrics,select_train_metrics,select_ev_metrics=set_value_for_dic_list(2,select_test_metrics,current_test_value,select_train_metrics,current_train_value,select_ev_metrics,current_ev_value)

    if (model_selection==3 or model_selection==5) and test_loss<select_test_metrics[0][3]:
      best_model[3]=deepcopy(model)
      select_test_metrics,select_train_metrics,select_ev_metrics=set_value_for_dic_list(3,select_test_metrics,current_test_value,select_train_metrics,current_train_value,select_ev_metrics,current_ev_value)

    if (model_selection==4 or model_selection==5) and ev_acc>select_ev_metrics[1][4]:
      best_model[4]=deepcopy(model)
      select_test_metrics,select_train_metrics,select_ev_metrics=set_value_for_dic_list(4,select_test_metrics,current_test_value,select_train_metrics,current_train_value,select_ev_metrics,current_ev_value)

  if (model_selection==0 or model_selection==5):
    print("#####select with train_acc#####")
    display(0,select_test_metrics,select_train_metrics,select_ev_metrics)
  if (model_selection==1 or model_selection==5):
    print("#####select with test_acc#####")
    display(1,select_test_metrics,select_train_metrics,select_ev_metrics)
  if (model_selection==2 or model_selection==5):
    print("#####select with train_loss#####")
    display(2,select_test_metrics,select_train_metrics,select_ev_metrics)
  if (model_selection==3 or model_selection==5):
    print("#####select with test_loss#####")
    display(3,select_test_metrics,select_train_metrics,select_ev_metrics)
  if (model_selection==4 or model_selection==5):
    print("#####select with ev_acc#####")
    display(4,select_test_metrics,select_train_metrics,select_ev_metrics)
  for model_s_k in best_model.keys():
    if best_model[model_s_k]==None:
      print("None value of the key:")
      print(model_s_k)
    best_model[model_s_k]=best_model[model_s_k].to(torch.device('cpu'))

  #return best_model,ev_acc,ev_pre,ev_rec
  #model, acc, pre, rec, auc, f1, low_fpr_tpr
  return best_model,select_ev_metrics[1],select_ev_metrics[2],select_ev_metrics[3],select_ev_metrics[4],select_ev_metrics[5],select_ev_metrics[10]


def train_and_ev_attack_model(t_all_data,t_ev_data,model_selection):
  """
  all_data (pd.DataFrame): training and testing data for attack model training, from shadow dataset
  ev_data (pd.DataFrame): attack features of target dataset
  """
  all_data=deepcopy(t_all_data)
  ev_data=deepcopy(t_ev_data)
  x_label=[item for item in all_data.columns if item != 'label' and item != 'o_index']

  X_train, X_test, y_train, y_test = train_test_split(all_data[x_label],all_data['label'], test_size=0.2) #, ran

  # print("*********Neural Network********")
  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print("attack model training device:")
  print(device)
  attack_lr=0.001#0.001#1e-4#0.0001
  attack_batch_size=32#32#8
  attack_epoch_num=300
  attack_model=model_attack(input_n=len(x_label),hidden_n=64,output_n=1).to(device)
  attack_criterion=torch.nn.BCELoss()

  attack_optimizer=optim.Adam(params=attack_model.parameters(),lr=attack_lr)#,weight_decay=1e-7
  x_list=x_label
  y_list=['label']
  print("x_list:%s"%(str(x_list)))
  train_dataset=pd.concat([X_train,y_train],axis=1)
  # print(train_dataset['label'])
  test_dataset=pd.concat([X_test,y_test],axis=1)
  scheduler=None
  attack_model,select_ev_acc,select_ev_pre,select_ev_rec,select_ev_auc,select_ev_f1,select_ev_low_fpr_tpr=train_one_attack_dataset(train_dataset,test_dataset,ev_data,attack_epoch_num,attack_batch_size,attack_model,x_list,y_list,device,attack_optimizer,scheduler,attack_criterion,model_selection)
  return attack_model,select_ev_acc,select_ev_pre,select_ev_rec,select_ev_auc,select_ev_f1,select_ev_low_fpr_tpr

if __name__=='__main__':
  test_file='test.csv'
  train_file='train.csv'
  df_train=pd.read_csv(train_file,index_col=0)
  df_test=pd.read_csv(test_file,index_col=0)

  df_all_data=pd.concat([df_train,df_test],ignore_index=True,axis=0)
  df_all_data=df_all_data.sample(frac=1).reset_index(drop=True) #shuffle and reset index

  attack_model,select_ev_acc,select_ev_pre,select_ev_rec=train_and_ev_attack_model(df_all_data,df_all_data)