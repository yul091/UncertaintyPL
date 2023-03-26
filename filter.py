import sys
sys.dont_write_bytecode = True
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
from BasicalClass.common_function import *
from program_tasks.code_summary.CodeLoader import CodeLoader
from program_tasks.code_summary.main import (
    perpare_train, 
    my_collate, 
    test_model, 
    dict2list,
)
from program_tasks.code_completion.vocab import VocabBuilder
from program_tasks.code_completion.dataloader import Word2vecLoader
from program_tasks.code_completion.main import test
from preprocess.checkpoint import Checkpoint
from sklearn.metrics import roc_curve, auc, brier_score_loss, precision_recall_curve


class Filter:
    def __init__(
        self, 
        res_dir: str, 
        data_dir: str, 
        metric_dir: str, 
        save_dir: str, 
        device: torch.device, 
        module_id: int, 
        shift: str, 
        max_size: int, 
        batch_size: int,
    ):

        self.res_dir = res_dir
        self.data_dir = data_dir
        self.device = device
        self.shift = shift
        self.module_id = module_id
        self.embed_type = 1
        self.vec_path = None
        self.embed_dim = 120
        self.max_size = max_size
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.vanilla = torch.load(os.path.join(metric_dir, 'Vanilla.res'))
        self.temp = torch.load(os.path.join(metric_dir, 'ModelWithTemperature.res'))
        self.pv = torch.load(os.path.join(metric_dir, 'PVScore.res'))
        self.dropout = torch.load(os.path.join(metric_dir, 'ModelActivateDropout.res'))
        self.mutation = torch.load(os.path.join(metric_dir, 'Mutation.res'))
        self.truth = torch.load(os.path.join(metric_dir, 'truth.res'))
        self.batch_size = batch_size

        if module_id == 0: # code summary
            self.tk_path = os.path.join(self.data_dir, 'tk.pkl')
            self.train_path = os.path.join(self.data_dir, 'train.pkl')
            self.val_path = os.path.join(self.data_dir, 'val.pkl')
            if shift:
                self.test_path = None
                self.test1_path = os.path.join(self.data_dir, 'test1.pkl')
                self.test2_path = os.path.join(self.data_dir, 'test2.pkl')
                self.test3_path = os.path.join(self.data_dir, 'test3.pkl')
            else:
                self.test_path = os.path.join(self.data_dir, 'test.pkl')
                self.test1_path = None
                self.test2_path = None
                self.test3_path = None
        else: # code completion
            self.tk_path = None
            self.train_path = os.path.join(self.data_dir, 'train.tsv')
            self.val_path = os.path.join(self.data_dir, 'val.tsv')
            if shift:
                self.test_path = None
                self.test1_path = os.path.join(self.data_dir, 'test1.tsv')
                self.test2_path = os.path.join(self.data_dir, 'test2.tsv')
                self.test3_path = os.path.join(self.data_dir, 'test3.tsv')
            else:
                self.test_path = os.path.join(self.data_dir, 'test.tsv')
                self.test1_path = None
                self.test2_path = None
                self.test3_path = None

        # load ckpt 
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        self.model = resume_checkpoint.model
        self.model.to(self.device)
        self.model.eval()

        if module_id == 0: # code summary
            # load data and preparation
            self.token2index, path2index, func2index, embed, self.tk2num = perpare_train(
                self.tk_path, self.embed_type, self.vec_path, 
                self.embed_dim, self.res_dir
            )
            self.index2func = dict2list(func2index)
            
            # build test loader
            if shift:
                self.test_dataset1 = CodeLoader(
                    self.test1_path, self.max_size, self.token2index, self.tk2num
                )
                self.test_dataset2 = CodeLoader(
                    self.test2_path, self.max_size, self.token2index, self.tk2num
                )
                self.test_dataset3 = CodeLoader(
                    self.test3_path, self.max_size, self.token2index, self.tk2num
                )
            else:
                self.test_dataset = CodeLoader(
                    self.test_path, self.max_size, self.token2index, self.tk2num
                )
        else: # code completion
            min_samples = 5
            # load data and preparation
            v_builder = VocabBuilder(path_file=self.train_path)
            self.d_word_index, embed = v_builder.get_word_index(min_sample=min_samples)

            if embed is not None:
                if type(embed) is np.ndarray:
                    embed = torch.tensor(embed, dtype=torch.float).cuda()
                assert embed.size()[1] == self.embed_dim
                
                
    def common_get_auc(self, y_test, y_score):
        # calculate true positive & false positive
        try:
            fpr, tpr, threshold = roc_curve(y_test, y_score)  
            roc_auc = auc(fpr, tpr)  # calculate AUC
            return roc_auc 
        except:
            return 0.0

    def common_get_aupr(self, y_test, y_score):
        try:
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)
            area = auc(recall, precision)
            return area
        except:
            return 0.0

    def common_get_brier(self, y_test, y_score):
        try:
            brier = brier_score_loss(y_test, y_score)
            return brier
        except:
            return 1.0
        
        
    def get_filtered_stats(
        self, 
        UE_scores: dict,
        UE_name: str,  
        testset: str, 
        threshold: float, 
        data_path: str, 
        original_size: int,
        res: dict,
    ):
        if UE_name in ['mutation', 'dissector']:
            scores = UE_scores[testset][0]
        else:
            scores = UE_scores[testset]
        idx = np.where(scores >= threshold)[0]
        coverage = len(idx) / original_size
        auc = self.common_get_auc(
            self.truth[testset][idx], 
            scores[idx],
        )
        aupr = self.common_get_aupr(
            self.truth[testset][idx], 
            scores[idx],
        )
        brier = self.common_get_brier(
            self.truth[testset][idx], 
            scores[idx],
        )
        if self.module_id == 0: # code summary
            dataset = CodeLoader(
                data_path, 
                self.max_size, 
                self.token2index, 
                self.tk2num, 
                idx=idx,
            )
            test_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                collate_fn=my_collate,
            )
            acc = test_model(
                test_loader, 
                self.model, 
                self.device, 
                self.index2func, 
                testset,
            )[f'{testset} acc']
        else: # code completion
            test_loader = Word2vecLoader(
                data_path, 
                self.d_word_index, 
                batch_size=self.batch_size, 
                max_size=self.max_size, 
                idx=list(idx),
            )
            acc = test(test_loader, self.model, testset)[f'{testset} acc']
        
        res[UE_name][testset]['coverage'].append(coverage)
        res[UE_name][testset]['F-1'].append(acc)
        res[UE_name][testset]['AUC'].append(auc)
        res[UE_name][testset]['AUPR'].append(aupr)
        res[UE_name][testset]['Brier'].append(brier)
        
        print('{} set [{} w/ threshold {}]: coverage {:.2f}, F-1 {:.2f}, AUC {:.2f}, AUPR {:.2f}, Brier {:.2f}'.format(
            testset, UE_name, threshold, coverage, acc, auc, aupr, brier,
        ))


    def filtering(self, res, threshold, testset='val'):
        original_size = len(self.vanilla[testset])
        # remain_size = int(coverage * original_size)
        if testset == 'val':
            data_path = self.val_path
        elif testset == 'test1':
            data_path = self.test1_path
        elif testset == 'test2':
            data_path = self.test2_path
        elif testset == 'test3':
            data_path = self.test3_path
        elif testset == 'test':
            data_path = self.test_path
        else:
            raise ValueError()

        # Vanilla
        # va_idx = np.argsort(self.vanilla[testset])[::-1][:remain_size]
        self.get_filtered_stats(
            UE_scores=self.vanilla,
            UE_name='vanilla',
            testset=testset,
            threshold=threshold,
            data_path=data_path,
            original_size=original_size,
            res=res,
        )
        
        # Temperature scaling
        # temp_idx = np.argsort(self.temp[testset])[::-1][:remain_size]
        self.get_filtered_stats(
            UE_scores=self.temp,
            UE_name='temperature',
            testset=testset,
            threshold=threshold,
            data_path=data_path,
            original_size=original_size,
            res=res,
        )

        # Mutation
        # mutation_idx = np.argsort(self.mutation[testset][0])[::-1][:remain_size]
        self.get_filtered_stats(
            UE_scores=self.mutation,
            UE_name='mutation',
            testset=testset,
            threshold=threshold,
            data_path=data_path,
            original_size=original_size,
            res=res,
        )

        # Dropout
        # dropout_idx = np.argsort(self.dropout[testset])[::-1][:remain_size]
        self.get_filtered_stats(
            UE_scores=self.dropout,
            UE_name='dropout',
            testset=testset,
            threshold=threshold,
            data_path=data_path,
            original_size=original_size,
            res=res,
        )

        # Dissector
        # pv_idx = np.argsort(self.pv[testset][0])[::-1][:remain_size]
        self.get_filtered_stats(
            UE_scores=self.pv,
            UE_name='dissector',
            testset=testset,
            threshold=threshold,
            data_path=data_path,
            original_size=original_size,
            res=res,
        )
        
        
    def get_coveraged_filtered_stats(
        self, 
        UE_scores: dict,
        UE_name: str,  
        testset: str, 
        coverage: float, 
        data_path: str, 
        remain_size: int,
        res: dict,
    ):
        if UE_name in ['mutation', 'dissector']:
            scores = UE_scores[testset][0]
        else:
            scores = UE_scores[testset]
            
        idx = np.argsort(scores)[::-1][:remain_size]
        threshold = scores[idx[-1]]
        auc = self.common_get_auc(
            self.truth[testset][idx], 
            scores[idx],
        )
        aupr = self.common_get_aupr(
            self.truth[testset][idx], 
            scores[idx],
        )
        brier = self.common_get_brier(
            self.truth[testset][idx], 
            scores[idx],
        )
        if self.module_id == 0: # code summary
            dataset = CodeLoader(
                data_path, 
                self.max_size, 
                self.token2index, 
                self.tk2num, 
                idx=idx,
            )
            test_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                collate_fn=my_collate,
            )
            acc = test_model(
                test_loader, 
                self.model, 
                self.device, 
                self.index2func, 
                testset,
            )[f'{testset} acc']
        else: # code completion
            test_loader = Word2vecLoader(
                data_path, 
                self.d_word_index, 
                batch_size=self.batch_size, 
                max_size=self.max_size, 
                idx=list(idx),
            )
            acc = test(test_loader, self.model, testset)[f'{testset} acc']
        
        res[UE_name][testset]['threshold'].append(threshold)
        res[UE_name][testset]['F-1'].append(acc)
        res[UE_name][testset]['AUC'].append(auc)
        res[UE_name][testset]['AUPR'].append(aupr)
        res[UE_name][testset]['Brier'].append(brier)
        
        print('{} set [{} w/ coverage {}]: threshold {:.2f}, F-1 {:.2f}, AUC {:.2f}, AUPR {:.2f}, Brier {:.2f}'.format(
            testset, UE_name, coverage, threshold, acc, auc, aupr, brier,
        ))

        
    def coverage_filtering(self, res, coverage, testset='val'):
        original_size = len(self.vanilla[testset])
        remain_size = int(coverage * original_size)
        if testset == 'val':
            data_path = self.val_path
        elif testset == 'test1':
            data_path = self.test1_path
        elif testset == 'test2':
            data_path = self.test2_path
        elif testset == 'test3':
            data_path = self.test3_path
        elif testset == 'test':
            data_path = self.test_path
        else:
            raise ValueError()

        # Vanilla
        self.get_coveraged_filtered_stats(
            UE_scores=self.vanilla,
            UE_name='vanilla',
            testset=testset,
            coverage=coverage,
            data_path=data_path,
            remain_size=remain_size,
            res=res,
        )
        
        # Temperature scaling
        self.get_coveraged_filtered_stats(
            UE_scores=self.temp,
            UE_name='temperature',
            testset=testset,
            coverage=coverage,
            data_path=data_path,
            remain_size=remain_size,
            res=res,
        )

        # Mutation
        self.get_coveraged_filtered_stats(
            UE_scores=self.mutation,
            UE_name='mutation',
            testset=testset,
            coverage=coverage,
            data_path=data_path,
            remain_size=remain_size,
            res=res,
        )

        # Dropout
        self.get_coveraged_filtered_stats(
            UE_scores=self.dropout,
            UE_name='dropout',
            testset=testset,
            coverage=coverage,
            data_path=data_path,
            remain_size=remain_size,
            res=res,
        )

        # Dissector
        self.get_coveraged_filtered_stats(
            UE_scores=self.pv,
            UE_name='dissector',
            testset=testset,
            coverage=coverage,
            data_path=data_path,
            remain_size=remain_size,
            res=res,
        )


    def run(self, strategy='threshold'):
        # evaluate on test dataset
        value_range = np.arange(0.01, 1.01, 0.01)
        if strategy == 'threshold':
            res = {
                'threshold': value_range, 
                'vanilla': {}, 
                'temperature': {}, 
                'mutation': {},
                'dropout': {},
                'dissector': {},
            }
        else:
            res = {
                'coverage': value_range, 
                'vanilla': {}, 
                'temperature': {}, 
                'mutation': {},
                'dropout': {},
                'dissector': {},
            }

        if self.shift:
            testsets = ['val', 'test1', 'test2', 'test3']
        else:
            testsets = ['val', 'test']
        for testset in testsets:
            res['vanilla'][testset] = defaultdict(list)
            res['temperature'][testset] = defaultdict(list)
            res['mutation'][testset] = defaultdict(list)
            res['dropout'][testset] = defaultdict(list)
            res['dissector'][testset] = defaultdict(list)
            for threshold in tqdm(value_range):
                if strategy == 'threshold':
                    self.filtering(res, threshold, testset)
                else:
                    self.coverage_filtering(res, threshold, testset)
            
        # Save file 
        if strategy == 'threshold':
            torch.save(res, os.path.join(self.save_dir, 'filter.res'))
        else:
            torch.save(res, os.path.join(self.save_dir, 'filter_coverage.res'))



if __name__ == "__main__":
    import warnings
    # Turn off all warnings
    warnings.filterwarnings("ignore")
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift_type', type=str, default='different_time',
                        choices=[
                            'different_time', 
                            'different_author', 
                            'different_project',
                            'case_study'
                        ],
                        help='Type of code distribution shift')
    parser.add_argument('--task', type=str, default='code_summary',
                        choices=['code_summary', 'code_completion'],
                        help='Type of program task')
    parser.add_argument('--model', type=str, default='codebert')
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--model_dir', type=str, default='results')
    parser.add_argument('--uncertainty_dir', type=str, default='Uncertainty_Results',
                        help='Directory where the uncertainty results are stored')
    parser.add_argument('--out_dir', type=str, default='Uncertainty_Eval/filter',
                        help='Directory where the filtering results are stored')
    parser.add_argument('--strategy', type=str, default='threshold',
                        choices=['threshold', 'coverage'],
                        help='Type of filtering strategy')
    
    args = parser.parse_args()
    task = args.task
    module_id = 0 if task == 'code_summary' else 1
    shift_type = args.shift_type
    model_type = args.model
    strategy = args.strategy
    dataset_dir = args.data_dir
    model_dir = args.model_dir
    uncertainty_dir = args.uncertainty_dir
    out_dir = args.out_dir
    module = 'CodeSummary_Module' if task == 'code_summary' else 'CodeCompletion_Module'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    res_dir = f'{model_dir}/{task}/{shift_type}/{model_type}'
    metric_dir = f'{uncertainty_dir}/{shift_type}/{model_type}/{module}'
    save_dir = f'{out_dir}/{shift_type}/{model_type}/{module}'
    if task == 'code_summary':
        data_dir = f'{dataset_dir}/{task}/{shift_type}/java_pkl'
    else:
        data_dir = f'{dataset_dir}/{task}/{shift_type}/java_project'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    shift = True if shift_type != 'case_study' else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_size = None if task == 'code_summary' else 200
    batch_size = 256 if task == 'code_summary' else 64

    filter = Filter(
        res_dir=res_dir, 
        data_dir=data_dir, 
        metric_dir=metric_dir,
        save_dir=save_dir, 
        device=device, 
        module_id=module_id, 
        shift=shift,
        max_size=max_size, 
        batch_size=batch_size,
    )

    filter.run(strategy=strategy)

    # Turn on warnings again
    warnings.filterwarnings("default")





