import sys
sys.dont_write_bytecode = True
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Union, Any
import torch
from BasicalClass.common_function import *


class Filter:
    def __init__(self, metric_dir: str, save_dir: str, shift: str):
        self.shift = shift
        self.module_id = module_id
        self.max_size = max_size
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.vanilla = torch.load(os.path.join(metric_dir, 'Vanilla.res'))
        self.temp = torch.load(os.path.join(metric_dir, 'ModelWithTemperature.res'))
        self.pv = torch.load(os.path.join(metric_dir, 'PVScore.res'))
        self.dropout = torch.load(os.path.join(metric_dir, 'ModelActivateDropout.res'))
        self.mutation = torch.load(os.path.join(metric_dir, 'Mutation.res'))
        # self.truth = torch.load(os.path.join(metric_dir, 'truth.res'))
        
        
    def get_filtered_stats(
        self, 
        UE_dict: Dict[str, Any],
        UE_name: str, 
        split: str, 
        original_size: int,
        res: Dict[str, Any],
        threshold: float = None,
        coverage: float = None,
        using_coverage: bool = False,
    ):
        all_scores = UE_dict[split] # List of scores
        for i, scores in enumerate(all_scores):
            if using_coverage:
                remain_size = int(coverage * original_size)
                idx = np.argsort(scores)[::-1][:remain_size]
                threshold = scores[idx[-1]]
            else:
                idx = np.where(scores >= threshold)[0]
                coverage = len(idx) / original_size
                
            if UE_name != 'mutation':
                truths = UE_dict[split]['truths'][0]
            else:
                truths = UE_dict[split]['truths'][i]
                
            auc = common_get_auc(truths[idx], scores[idx])
            aupr = common_get_aupr(truths[idx], scores[idx])
            brier = common_get_brier(truths[idx], scores[idx])
            acc = np.mean(truths[idx])
        
            res[UE_name][split]['coverage'].append(coverage)
            res[UE_name][split]['threshold'].append(threshold)
            res[UE_name][split]['F-1'].append(acc)
            res[UE_name][split]['AUC'].append(auc)
            res[UE_name][split]['AUPR'].append(aupr)
            res[UE_name][split]['Brier'].append(brier)
        
            print('[{} set | {} sub-method {} | threshold {}]: coverage {:.2f}, F-1 {:.2f}, AUC {:.2f}, AUPR {:.2f}, Brier {:.2f}'.format(
                split, UE_name, i, threshold, coverage, acc, auc, aupr, brier,
            ))


    def filtering(self, res, split='dev', threshold=None, coverage=None, use_coverage=False):
        original_size = len(self.vanilla[split])
        for method in ['vanilla', 'temperature', 'mutation', 'dropout', 'dissector']:
            self.get_filtered_stats(
                UE_dict=self.vanilla,
                UE_name=method,
                split=split,
                original_size=original_size,
                res=res,
                threshold=threshold,
                coverage=coverage,
                using_coverage=use_coverage,
            )


    def run(self, strategy='threshold'):
        # evaluate on test dataset
        value_range = np.arange(0.01, 1.01, 0.01)
        if strategy == 'threshold':
            threshold = value_range
            coverage = None
        else:
            threshold = None
            coverage = value_range
        
        res = {
            'threshold': threshold, 
            'coverage': coverage,
            'vanilla': {}, 
            'temperature': {}, 
            'mutation': {},
            'dropout': {},
            'dissector': {},
        }

        if self.shift:
            splits = ['dev', 'test1', 'test2', 'test3']
        else:
            splits = ['dev', 'test']
            
        for split in splits:
            res['vanilla'][split] = defaultdict(list)
            res['temperature'][split] = defaultdict(list)
            res['mutation'][split] = defaultdict(list)
            res['dropout'][split] = defaultdict(list)
            res['dissector'][split] = defaultdict(list)
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
    data_dir = f'{dataset_dir}/{task}/{shift_type}'
    
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





