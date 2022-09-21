import argparse

cmd_opt = argparse.ArgumentParser(description='default setting for experiments')

cmd_opt.add_argument('-root_path', type=str, default='./datasets', help='The dir path for saving the datasets')
cmd_opt.add_argument('-split_rate', type=float, default=0.5, help='The percentage of target dataset in same dataset with shadow dataset')
cmd_opt.add_argument('-repeat_time', type=int, default=10, help='The number of time for repeating')
cmd_opt.add_argument('-w_explain', type=bool, default=False, help='Whether explain the attack model')
cmd_opt.add_argument('-attack_type', type=int, default=2, help='0: just confidence vector attack (NN-based and metric-baseds); 1: just label-only; 2:both')
#cmd_opt.add_argument('-sample_way', type=int, default=1, help='1: class balance; 0: random split')
cmd_opt.add_argument('-sample_way', type=int, default=1, help='0: random; 1: all balanced; 2: target_train and shadow_train balanced;')
#for splitting all nodes in the graph into 4 subgraphs
cmd_opt.add_argument('-model_selection', type=int, default=1, help='different model selection strategies while training attack model')
#selecting attack model with different metric (train_acc 0, test_acc 1, train_loss 2, test_loss 3, the best 4, try all strategies 5)
cmd_opt.add_argument('-fe_list_0', type=int, default=0, help='1 use fe_list_0; 0 not use')
cmd_opt.add_argument('-fe_list_1', type=int, default=0, help='1 use fe_list_1; 0 not use')
cmd_opt.add_argument('-fe_list_2', type=int, default=1, help='1 use fe_list_2; 0 not use')
cmd_opt.add_argument('-fe_list_3', type=int, default=1, help='1 use fe_list_3; 0 not use')
cmd_opt.add_argument('-fe_list_4', type=int, default=1, help='1 use fe_list_4; 0 not use')


cmd_opt.add_argument('-target_dataset_name', type=str, default='cora_ml', help='The default name of target dataset')
cmd_opt.add_argument('-target_train_test_rate', type=float, default=0.5, help='The split rate of training data in target dataset')
cmd_opt.add_argument('-target_model_name', type=str, default='gcn', help='The name of target model')
cmd_opt.add_argument('-target_optimizer_name', type=str, default='Adam', help='The name of optimizer')
cmd_opt.add_argument('-target_learning_rate', type=float, default=0.003, help='The learning rate of target model')
cmd_opt.add_argument('-target_mom_or_wd', type=float, default=1e-7, help='1e-7(wd) 0.9(mom)The momentum of target model if SGD used, or the weight_decay of Adam')
cmd_opt.add_argument('-target_criterion', type=str, default='CrossEntropyLoss', help='The criterion of target model')
cmd_opt.add_argument('-target_batch_size', type=int, default=32, help='The batch size of target model')
cmd_opt.add_argument('-target_epoch', type=int, default=200, help='The epoch number of target model')

#parameters for GNN models
cmd_opt.add_argument('-target_hidden_channels', type=int, default=32, help='The hidden channels of the GNN model')
cmd_opt.add_argument('-target_num_layers', type=int, default=4, help='The number of layers in each GNN model')
cmd_opt.add_argument('-target_dropout', type=float, default=0.0, help='The dropout rate for the hidden represent of each layer')
cmd_opt.add_argument('-target_act', type=str, default='relu', help='The activation function for each layer')
cmd_opt.add_argument('-target_norm', type=str, default=None, help='The normalization for each layer')
cmd_opt.add_argument('-target_jk', type=str, default=None, help='The jumping knowledge configure')
cmd_opt.add_argument('-target_act_first', type=bool, default=False, help='The order of activation function with normalization')
cmd_opt.add_argument('-target_act_kwargs', type=dict, default=None, help='The parameters for activation function')
cmd_opt.add_argument('-target_readout', type=str, default=None, help='The method of aggregating features of nodes to the graph, for node classification, just None')

cmd_opt.add_argument('-shadow_dataset_name', type=str, default='cora_ml', help='The default name of shadow dataset')
cmd_opt.add_argument('-shadow_train_test_rate', type=float, default=0.5, help='The split rate of training data in shadow dataset')
cmd_opt.add_argument('-shadow_model_name', type=str, default='gcn', help='The name of shadow model')
cmd_opt.add_argument('-shadow_optimizer_name', type=str, default='Adam', help='The name of optimizer')
cmd_opt.add_argument('-shadow_learning_rate', type=float, default=0.003, help='The learning rate of shadow model')
cmd_opt.add_argument('-shadow_mom_or_wd', type=float, default=1e-7, help='The momentum of shadow model if SGD used, or the weight_decay of Adam')
cmd_opt.add_argument('-shadow_criterion', type=str, default='CrossEntropyLoss', help='The criterion of shadow model')
cmd_opt.add_argument('-shadow_batch_size', type=int, default=32, help='The batch size of shadow model')
cmd_opt.add_argument('-shadow_epoch', type=int, default=200, help='The epoch number of shadow model')

#parameters for GNN models
cmd_opt.add_argument('-shadow_hidden_channels', type=int, default=32, help='The hidden channels of the GNN model')
cmd_opt.add_argument('-shadow_num_layers', type=int, default=4, help='The number of layers in each GNN model')
cmd_opt.add_argument('-shadow_dropout', type=float, default=0.0, help='The dropout rate for the hidden represent of each layer')
cmd_opt.add_argument('-shadow_act', type=str, default='relu', help='The activation function for each layer')
cmd_opt.add_argument('-shadow_norm', type=str, default=None, help='The normalization for each layer')
cmd_opt.add_argument('-shadow_jk', type=str, default=None, help='The jumping knowledge configure')
cmd_opt.add_argument('-shadow_act_first', type=bool, default=False, help='The order of activation function with normalization')
cmd_opt.add_argument('-shadow_act_kwargs', type=dict, default=None, help='The parameters for activation function')
cmd_opt.add_argument('-shadow_readout', type=str, default=None, help='The method of aggregating features of nodes to the graph, for node classification, just None')


cmd_args, _ = cmd_opt.parse_known_args()