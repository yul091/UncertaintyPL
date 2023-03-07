import torch
import os
import numpy as np
from BasicalClass.common_function import *
from java_dataset.se_tasks.code_summary.scripts.Code2VecModule import Code2Vec
from java_dataset.se_tasks.code_summary.scripts.CodeLoader import CodeLoader
from java_dataset.se_tasks.code_summary.scripts.main import perpare_train, my_collate
from java_dataset.checkpoint import Checkpoint
from tqdm import tqdm
from collections import defaultdict

class InputVal(object):
    """
    Given a dataset, with the input uncertain scores, models as well as the expected prediction accuracy,
    will return the filtering threshold and coverage.
    Inputs:
        expt_acc (float): the expected prediction accuracy for the filtered dataset.
        device (object): torch.device("cuda" if torch.cuda.is_available() else "cpu").
    Outputs:
        threshold (list): list of the highest threshold for input filtering for each test set.
        coverage (list): list of coverage scores for each test set given the chosen threshold.
    """
    RES_DIR = 'java_dataset/se_tasks/code_summary/result'
    CHECKPOINT_DIR_NAME = 'checkpoints'
    MODEL_NAME = 'model_state.pt'
    UN_DIR = 'Uncertainty_Results/CodeSummary_Module'
    DATA_DIR = 'java_dataset/data/code_summary-preprocess'
    DATASETS = ['val', 'shift1', 'shift2']
    RES_FILE = 'input_val'

    def __init__(self, device, expt_acc, gap=0.01):
        self.tk_path = os.path.join(self.DATA_DIR, 'tk.pkl')
        self.train_path = os.path.join(self.DATA_DIR, 'train.pkl')
        self.shift1_path = os.path.join(self.DATA_DIR, 'shift1.pkl')
        self.shift2_path = os.path.join(self.DATA_DIR, 'shift2.pkl')
        self.val_path = os.path.join(self.DATA_DIR, 'val.pkl')
        self.vec_path = 'java_dataset/embedding_vec/100_2/Doc2VecEmbedding0.vec'
        self.embed_dim = 100
        self.out_dir = self.RES_DIR
        self.max_size = None # use all of the data
        self.embed_type = 1 # best (train from scratch)
        self.device = device
        self.expt_acc = expt_acc
        self.test_batch_size = 256
        self.device = device
        self.gap = gap

        assert isinstance(expt_acc, float)
        assert 0 <= expt_acc <=1
        # uncertainty inputs
        for file in os.listdir(self.UN_DIR):
            if file.endswith('res'):
                # print(file)
                if 'Dropout' in file:
                    self.dropout = torch.load(os.path.join(self.UN_DIR, file))
                elif 'Temperature' in file:
                    self.temp = torch.load(os.path.join(self.UN_DIR, file))
                elif 'Mutation' in file:
                    self.mutation = torch.load(os.path.join(self.UN_DIR, file))
                elif 'PVScore' in file:
                    self.pv = torch.load(os.path.join(self.UN_DIR, file))
                elif 'Viallina' in file:
                    self.va = torch.load(os.path.join(self.UN_DIR, file))
                elif 'Mahalanobis' in file:
                    self.mahala = torch.load(os.path.join(self.UN_DIR, file))
                else:
                    pass

        self.model = self.load_model().to(self.device) # get trained model
        self.val_thre_50 = defaultdict(float)
        self.val_thre_100 = defaultdict(float)
        self.val_thre_200 = defaultdict(float)
        self.open50 = True
        self.open100 = True
        self.open200 = True
        self.val_acc_0 = self.cal_acc('val')
        print(f'val acc_0: {self.val_acc_0}') 
        self.shift1_acc_0 = self.cal_acc('shift1')
        print(f'shift1 acc_0: {self.shift1_acc_0}')          
        self.shift2_acc_0 = self.cal_acc('shift2')
        print(f'shift2 acc_0: {self.shift2_acc_0}')
        
    
    def load_model(self):
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.RES_DIR)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        return model

    def load_data(self, dataset_name, select_index=None):
        token2index, path2index, func2index, embed, tk2num = \
            perpare_train(
                self.tk_path, self.embed_type, self.vec_path, 
                self.embed_dim, self.out_dir)

        if dataset_name == 'train':
            return CodeLoader(self.train_path, self.max_size, token2index, tk2num, select_index=select_index)
        elif dataset_name == 'val':
            return CodeLoader(self.val_path, self.max_size, token2index, tk2num, select_index=select_index)
        elif dataset_name == 'shift1':
            return CodeLoader(self.shift1_path, self.max_size, token2index, tk2num, select_index=select_index)
        elif dataset_name == 'shift2':
            return CodeLoader(self.shift2_path, self.max_size, token2index, tk2num, select_index=select_index)
        else:
            raise TypeError('unsupported dataset type!')
        
        # print(f'train data length: {len(train_db)}, val length: {len(val_db)}, shift1 length: {len(shift1_db)}, shift2 length: {len(shift2_db)}')
        # return train_db, shift1_db, shift2_db, val_db

    def cal_acc(self, data_name):
        new_dataset = self.load_data(data_name)
        # evaluate
        new_loader = DataLoader(
            new_dataset, batch_size=self.test_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        pred_pos, pred_y, y = common_predict(new_loader, self.model, self.device)
        acc_0 = common_cal_accuracy(pred_y, y)
        return acc_0    

    def get_acc(self, threshold, uncertainty_score, data_name, uncertainty_name):
        # sort_index = uncertainty_score.argsort()[::-1] # reverse sort
        # remained_index = sort_index[:int(threshold*len(sort_index))] # larger scores remained
        remained_index = np.where(uncertainty_score >= threshold)[0]

        if len(remained_index) == len(uncertainty_score):
            print('No data filtered at this threshold %.4f!' %threshold)
            coverage = 1
            return None, coverage
        else:
            new_dataset = self.load_data(data_name, remained_index)
            print(f'{len(uncertainty_score)-len(new_dataset)} of data filtered!')
            if len(new_dataset) == 0:
                return 0.0, 0.0
            coverage = len(new_dataset) / len(uncertainty_score)
            # evaluate
            new_loader = DataLoader(
                new_dataset, batch_size=self.test_batch_size, 
                collate_fn=my_collate, shuffle=False
            )
            pred_pos, pred_y, y = common_predict(new_loader, self.model, self.device)
            acc = common_cal_accuracy(pred_y, y)

            if data_name == 'val':
                improve_level = (acc - self.val_acc_0)/self.val_acc_0
                print('threshold: %.4f, acc: %.4f, improve level: %.4f, coverage: %.4f'%(
                    threshold, acc, improve_level, coverage
                ))
                if  improve_level >= 0.5 and self.open50:
                    print(f'store threshold {threshold} to thre@50 for {uncertainty_name}')
                    self.val_thre_50[uncertainty_name] = threshold
                    self.open50 = False
                if  improve_level >= 1 and self.open100:
                    print(f'store threshold {threshold} to thre@100 for {uncertainty_name}')
                    self.val_thre_100[uncertainty_name] = threshold
                    self.open100 = False
                if  improve_level >= 2 and self.open200:
                    print(f'store threshold {threshold} to thre@200 for {uncertainty_name}')
                    self.val_thre_200[uncertainty_name] = threshold
                    self.open200 = False
            elif data_name == 'shift1':
                improve_level = (acc - self.shift1_acc_0)/self.shift1_acc_0
                print('threshold: %.4f, acc: %.4f, improve level: %.4f, coverage: %.4f'%(
                    threshold, acc, improve_level, coverage
                ))
            else:
                improve_level = (acc - self.shift2_acc_0)/self.shift2_acc_0
                print('threshold: %.4f, acc: %.4f, improve level: %.4f, coverage: %.4f'%(
                    threshold, acc, improve_level, coverage
                ))

            return acc, coverage

    def search_coverage(self, data_name, uncertainty, uncertainty_name, txt_file):
        # print('Uncertainty score: ', np.array(uncertainty[data_name]))
        record = []
        if data_name == 'val':
            for threshold in torch.arange(0, 1, self.gap):
                threshold = round(threshold.item(), 4)
                if uncertainty_name not in ['mutation', 'dissactor']:
                    acc, coverage = self.get_acc(threshold, uncertainty[data_name], data_name, uncertainty_name)
                else:
                    acc, coverage = self.get_acc(threshold, uncertainty[data_name][0], data_name, uncertainty_name)
                record.append([acc, coverage])

                if not (self.open50 or self.open100 or self.open200): # all the three thresholds has been stored
                    break
            
        else:
            threshold_50 = self.val_thre_50[uncertainty_name]
            threshold_100 = self.val_thre_100[uncertainty_name]
            threshold_200 = self.val_thre_200[uncertainty_name]
            if uncertainty_name not in ['mutation', 'dissactor']:
                acc_50, coverage_50 = self.get_acc(threshold_50, uncertainty[data_name], data_name, uncertainty_name)
                acc_100, coverage_100 = self.get_acc(threshold_100, uncertainty[data_name], data_name, uncertainty_name)
                acc_200, coverage_200 = self.get_acc(threshold_200, uncertainty[data_name], data_name, uncertainty_name)
            else:
                acc_50, coverage_50 = self.get_acc(threshold_50, uncertainty[data_name][0], data_name, uncertainty_name)
                acc_100, coverage_100 = self.get_acc(threshold_100, uncertainty[data_name][0], data_name, uncertainty_name)
                acc_200, coverage_200 = self.get_acc(threshold_200, uncertainty[data_name][0], data_name, uncertainty_name)
                # acc, coverage = self.get_acc(threshold, uncertainty[data_name][0], data_name, uncertainty_name)
            record.append([
                acc_50, coverage_50,
                acc_100, coverage_100,
                acc_200, coverage_200, 
            ])

        return record

    def calculate_coverage(self):
        # save all results to txt
        txt_file = open(os.path.join(self.UN_DIR, self.RES_FILE+'_'+str(self.expt_acc)+'.txt'), "w")
        dropout_record = defaultdict(list)
        vanilla_record = defaultdict(list)
        temp_record = defaultdict(list)
        mutation_record = defaultdict(list)
        dissactor_record = defaultdict(list)
        mahala_record = defaultdict(list)

        self.model.eval()
        
        for data_name in self.DATASETS:
            # if data_name == 'val' and os.path.exists(os.path.join(self.UN_DIR, 'threshold_50improve.txt')):
            #     self.val_thre_50 = torch.load(os.path.join(self.UN_DIR, 'threshold_50improve.txt'))
            #     self.val_thre_100 = torch.load(os.path.join(self.UN_DIR, 'threshold_100improve.txt'))
            #     self.val_thre_200 = torch.load(os.path.join(self.UN_DIR, 'threshold_200improve.txt'))
            #     continue

            print(f'dataset: {data_name}')
            #dropout
            if hasattr(self, 'dropout'):
                print('dropout:')
                txt_file.write('dropout_'+data_name+':\n')
                self.open50 = True
                self.open100 = True
                self.open200 = True
                dropout_record[data_name] = self.search_coverage(data_name, self.dropout, 'dropout', txt_file)
                
            # vanilla
            if hasattr(self, 'va'):
                print('vanilla:')
                txt_file.write('vanilla_'+data_name+':\n')
                self.open50 = True
                self.open100 = True
                self.open200 = True
                vanilla_record[data_name] = self.search_coverage(data_name, self.va, 'vanilla', txt_file)
                
            # mutation
            if hasattr(self, 'mutation'):
                print('mutation:')
                txt_file.write('mutation:'+data_name+':\n')
                self.open50 = True
                self.open100 = True
                self.open200 = True
                mutation_record[data_name] = self.search_coverage(data_name, self.mutation, 'mutation', txt_file)
                
            # dissactor
            if hasattr(self, 'pv'):
                print('dissactor:')
                txt_file.write('dissactor:'+data_name+':\n')
                self.open50 = True
                self.open100 = True
                self.open200 = True
                dissactor_record[data_name] = self.search_coverage(data_name, self.pv, 'dissactor', txt_file)
                
            # temp scale
            if hasattr(self, 'temp'):
                print('temp scale:')
                txt_file.write('temp scale:'+data_name+':\n')
                self.open50 = True
                self.open100 = True
                self.open200 = True
                temp_record[data_name] = self.search_coverage(data_name, self.temp, 'temp scale', txt_file)
                
            # mahalanobis
            if hasattr(self, 'mahala'):
                print('mahalanobis:')
                txt_file.write('mahalanobis:'+data_name+':\n')
                self.open50 = True
                self.open100 = True
                self.open200 = True
                mahala_record[data_name] = self.search_coverage(data_name, self.mahala, 'mahalanobis', txt_file)

            if data_name == 'val':
                torch.save(self.val_thre_50, os.path.join(self.UN_DIR, 'threshold_50improve.txt'))
                torch.save(self.val_thre_100, os.path.join(self.UN_DIR, 'threshold_100improve.txt'))
                torch.save(self.val_thre_200, os.path.join(self.UN_DIR, 'threshold_200improve.txt'))

        txt_file.close()
        torch.save(dropout_record, os.path.join(self.UN_DIR, 'dropout_record.txt'))
        torch.save(vanilla_record, os.path.join(self.UN_DIR, 'vanilla_record.txt'))
        torch.save(mutation_record, os.path.join(self.UN_DIR, 'mutation_record.txt'))
        torch.save(dissactor_record, os.path.join(self.UN_DIR, 'dissactor_record.txt'))
        torch.save(temp_record, os.path.join(self.UN_DIR, 'temp_record.txt'))
        torch.save(mahala_record, os.path.join(self.UN_DIR, 'mahala_record.txt'))

        


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expt_acc = 0.6
    # print(f'expected acc: {expt_acc}')
    input_val = InputVal(device, expt_acc, gap=0.01)
    input_val.calculate_coverage()
            
        
                
