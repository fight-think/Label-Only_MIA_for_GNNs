from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,precision_score,recall_score,f1_score
from bisect import bisect_left

def select_threshold_and_evaluate(shadow_data, target_data):
  """
  metric_key=['all_cor','p_o_g_t','cro_entropy','modified_cro_entropy'] label=['label'] to infer the membership
  shadow_data: DataFrame
  target_data: DataFrame
  
  """
  metric_key=['all_cor','p_o_g_t','cro_entropy','modified_cro_entropy']
  res_dict={}

  for metric in metric_key:
    res_dict[metric]=[]

    shadow_y=shadow_data['label'].tolist()
    shadow_x=shadow_data[metric].tolist()
    target_y=target_data['label'].tolist()
    target_x=target_data[metric].tolist()
    
    if metric == 'all_cor':
      ev_acc=accuracy_score(target_y,target_x)
      ev_pre=precision_score(target_y,target_x,zero_division=0)
      ev_rec=recall_score(target_y,target_x,zero_division=0)
      ev_f1=f1_score(target_y,target_x,zero_division=0)

      #[select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1]
      res_dict[metric]=[ev_acc,ev_pre,ev_rec,-1.0,ev_f1,-1.0]
      print("metric:"+metric+" res--ev_acc:%f, ev_pre:%f, ev_re:%f, ev_auc:%f, ev_f1:%f, ev_low_fpr_tpr:%f"%(ev_acc,ev_pre,ev_rec,-1.0,ev_f1,-1.0))
    
    else:
      #determine the threshold
      fpr, tpr, thresholds = roc_curve(shadow_y, shadow_x, pos_label=1)
      max_acc=0
      max_acc_pre=0
      max_acc_rec=0
      max_acc_f1=0
      max_acc_threshold=0
      for thres in thresholds:
        #evaluate on shadow data
        if metric in ['cro_entropy']:
          pre_as_thre=[1 if m < thres else 0 for m in shadow_x]
        elif metric in ['p_o_g_t','modified_cro_entropy']:
          pre_as_thre=[1 if m > thres else 0 for m in shadow_x]
        
        ev_acc=accuracy_score(shadow_y,pre_as_thre)
        if ev_acc>max_acc:
          max_acc=ev_acc
          max_acc_pre=precision_score(shadow_y,pre_as_thre,zero_division=0)
          max_acc_rec=recall_score(shadow_y,pre_as_thre,zero_division=0)
          max_acc_f1=f1_score(shadow_y,pre_as_thre,zero_division=0)
          max_acc_threshold=thres
      print("Selection acc:%f, pre:%f, re:%f, f1:%f"%(max_acc,max_acc_pre,max_acc_rec,max_acc_f1))
        
      #evaluate on target data
      if metric in ['cro_entropy']:
        pre_as_thre_target=[1 if m < max_acc_threshold else 0 for m in target_x]
      elif metric in ['p_o_g_t','modified_cro_entropy']:
        pre_as_thre_target=[1 if m > max_acc_threshold else 0 for m in target_x]

      #ev_auc=roc_auc_score(_true,ev_prob)
      ev_auc=roc_auc_score(target_y,target_x)
      ev_acc=accuracy_score(target_y,pre_as_thre_target)
      ev_pre=precision_score(target_y,pre_as_thre_target,zero_division=0)
      ev_rec=recall_score(target_y,pre_as_thre_target,zero_division=0)
      ev_f1=f1_score(target_y,pre_as_thre_target,zero_division=0)

      fpr, tpr, thresholds = roc_curve(target_y, target_x)
      fp_list=[0.01]
      tp_list=[]
      for item in fp_list:
        fp_index=bisect_left(fpr,item)
        if fp_index==0:
          tp_list.append(tpr[0])
        elif fp_list==len(fpr):
          raise ValueError("Give fpr larger than 1.0")
        else:
          tp_list.append(tpr[fp_index-1])
      ev_low_fpr_tpr=tp_list[0]

      #[select_ev_acc_1,select_ev_pre_1,select_ev_rec_1,select_ev_auc_1,select_ev_f1_1,select_ev_low_fpr_tpr_1]
      res_dict[metric]=[ev_acc,ev_pre,ev_rec,ev_auc,ev_f1,ev_low_fpr_tpr]
      print("metric:"+metric+" res--ev_acc:%f, ev_pre:%f, ev_re:%f, ev_auc:%f, ev_f1:%f, ev_low_fpr_tpr:%f"%(ev_acc,ev_pre,ev_rec,ev_auc,ev_f1,ev_low_fpr_tpr))
  return res_dict['all_cor'],res_dict['p_o_g_t'],res_dict['cro_entropy'],res_dict['modified_cro_entropy']

