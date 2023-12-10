import os
import torch
import numpy as np
import argparse
from BasicalClass.common_function import common_get_auc, common_get_aupr, common_get_brier


class Uncertainty_Eval():
    def __init__(self, res_dir, save_dir, task='CodeSummary_Module', shift=False, ood=False):
        """
        res_dir (str): path of uncertainty result, Default: "Uncertainty_Results".
        save_dir (str): path of saving evaluation res, Default: "Uncertainty_Eval/java".
        task (str): task name like CodeSummary_Module.
        """
        self.res_dir = res_dir
        self.task = task
        self.save_dir = save_dir
        self.shift = shift
        self.ood = ood
    

    def common_cal(self, y_test, y_score, metric='AUC'):
        if metric.lower() == 'auc':
            return common_get_auc(y_test, y_score)
        elif metric.lower() == 'aupr':
            return common_get_aupr(y_test, y_score)
        elif metric.lower() == 'brier':
            return common_get_brier(y_test, y_score)
        else:
            raise TypeError("Unknown metric type!")
        

    def cal_mUncertainty(self, metric_name, metric_res, eval_res):
        mU_vals = [np.mean(res) for res in metric_res['dev']['UE_scores']]
        if self.ood:
            mU_oods = [np.mean(res) for res in metric_res['ood']['UE_scores']]
        if not self.shift:
            mU_tests = [np.mean(res) for res in metric_res['test']['UE_scores']]
            count = 0
            if not self.ood:
                for mU_val, mU_test in zip(mU_vals, mU_tests):
                    count += 1
                    print('%s (method %d): \nmUncertainty: val: %.4f, test: %.4f' % (
                        metric_name, count, mU_val, mU_test
                    ))
                eval_res[metric_name]['mUncertain'] = [
                    {'dev': mU_val, 'test': mU_test}
                    for mU_val, mU_test in zip(mU_vals, mU_tests)
                ]
            else:
                for mU_val, mU_test, mU_ood in zip(mU_vals, mU_tests, mU_oods):
                    count += 1
                    print('%s (method %d): \nmUncertainty: val: %.4f, test: %.4f, ood: %.4f' % (
                        metric_name, count, mU_val, mU_test, mU_ood
                    ))
                eval_res[metric_name]['mUncertain'] = [
                    {'dev': mU_val, 'test': mU_test, 'ood': mU_ood}
                    for mU_val, mU_test in zip(mU_vals, mU_tests, mU_oods)
                ]
        else:
            mU1s = [np.mean(res) for res in metric_res['test1']['UE_scores']]
            mU2s = [np.mean(res) for res in metric_res['test2']['UE_scores']]
            mU3s = [np.mean(res) for res in metric_res['test3']['UE_scores']]
            count = 0
            if not self.ood:
                for mU_val, mU1, mU2, mU3 in zip(mU_vals, mU1s, mU2s, mU3s):
                    count += 1
                    print('%s (method %d): \nmUncertainty: val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f' % (
                        metric_name, count, mU_val, mU1, mU2, mU3
                    ))
                eval_res[metric_name]['mUncertain'] = [
                    {'dev': mU_val, 'test1': mU1, 'test2': mU2, 'test3': mU3}
                    for mU_val, mU1, mU2, mU3 in zip(mU_vals, mU1s, mU2s, mU3s)
                ]
            else:
                for mU_val, mU1, mU2, mU3, mU_ood in zip(mU_vals, mU1s, mU2s, mU3s, mU_oods):
                    count += 1
                    print('%s (method %d): \nmUncertainty: val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f, ood: %.4f' % (
                        metric_name, count, mU_val, mU1, mU2, mU3, mU_ood
                    ))
                eval_res[metric_name]['mUncertain'] = [
                    {'dev': mU_val, 'test1': mU1, 'test2': mU2, 'test3': mU3, 'ood': mU_ood}
                    for mU_val, mU1, mU2, mU3, mU_ood in zip(mU_vals, mU1s, mU2s, mU3s, mU_oods)
                ]


    def cal_metric(self, metric_name, metric_res, eval_res, metric='AUC'):
        metric_vals = [
            self.common_cal(metric_res['dev']['truths'][0], scores, metric) for scores in metric_res['dev']['UE_scores']
        ]
        if not self.shift:
            metric_tests = [
                self.common_cal(metric_res['test']['truths'][0], scores, metric) for scores in metric_res['test']['UE_scores']
            ]
            count = 0
            for metric_val, metric_test in zip(metric_vals, metric_tests):
                count += 1
                print('%s (method %d): val: %.4f, test: %.4f' % (
                    metric, count, metric_val, metric_test
                ))
            eval_res[metric_name][metric] = [
                {'dev': metric_val, 'test': metric_test} 
                for metric_val, metric_test in zip(metric_vals, metric_tests)
            ]
        else:
            mts1 = [
                self.common_cal(metric_res['test1']['truths'][0], res, metric) 
                for res in metric_res['test1']['UE_scores']
            ]
            mts2 = [
                self.common_cal(metric_res['test2']['truths'][0], res, metric) 
                for res in metric_res['test2']['UE_scores']
            ]
            mts3 = [
                self.common_cal(metric_res['test3']['truths'][0], res, metric) 
                for res in metric_res['test3']['UE_scores']
            ]
            count = 0
            for metric_val, mt1, mt2, mt3 in zip(metric_vals, mts1, mts2, mts3):
                count += 1
                print('%s (method %d): val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f' % (
                    metric, count, metric_val, mt1, mt2, mt3
                ))
            eval_res[metric_name][metric] = [
                {'dev': metric_val, 'test1': mt1, 'test2': mt2, 'test3': mt3}
                for metric_val, mt1, mt2, mt3 in zip(metric_vals, mts1, mts2, mts3)
            ]


    def evaluation(self):
        eval_res = {}
        src_dir = os.path.join(self.res_dir, self.task)
        truth = torch.load(os.path.join(src_dir,'truth.res'))
        uncertainty_res = [f for f in os.listdir(src_dir) if f.endswith('.res') and f != 'truth.res']
        
        if not self.shift:
            if not self.ood:
                print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f' % (
                    np.mean(truth['train']), np.mean(truth['dev']), np.mean(truth['test'])
                ))
            else:
                print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f, ood_acc: %.4f' % (
                    np.mean(truth['train']), np.mean(truth['dev']), 
                    np.mean(truth['test']), np.mean(truth['ood'])
                ))
        for metric in uncertainty_res:
            metric_res = torch.load(os.path.join(src_dir, metric))
            metric_name = metric[:-4] # get rid of endswith '.res'
            eval_res[metric_name] = {}

            # average uncertainty
            self.cal_mUncertainty(metric_name, metric_res, eval_res)
            # AUC
            self.cal_metric(metric_name, metric_res, eval_res, 'AUC')
            # AUPR
            self.cal_metric(metric_name, metric_res, eval_res, 'AUPR')
            # Brier score
            self.cal_metric(metric_name, metric_res, eval_res, 'Brier')

        # save evaluation res
        if not os.path.exists(os.path.join(self.save_dir, self.task)):
            os.makedirs(os.path.join(self.save_dir, self.task))
        save_name = os.path.join(self.save_dir, self.task, 'uncertainty_eval.res')
        torch.save(eval_res, save_name)

    
    def ood_detect(self):
        ood_res = {}
        src_dir = os.path.join(self.res_dir, self.task)
        truth = torch.load(os.path.join(src_dir,'truth.res'))
        uncertainty_res = [f for f in os.listdir(src_dir) if f.endswith('.res') and f != 'truth.res']
        
        if not self.shift:
            print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f, ood_acc: %.4f' % (
                np.mean(truth['train']), np.mean(truth['dev']), 
                np.mean(truth['test']), np.mean(truth['ood'])
            ))
        else:
            print('train_acc: %.4f, val_acc: %.4f, test1_acc: %.4f, test2_acc: %.4f, test3_acc: %.4f, ood_acc: %.4f' % (
                np.mean(truth['train']), np.mean(truth['dev']), 
                np.mean(truth['test1']), np.mean(truth['test2']), 
                np.mean(truth['test3']), np.mean(truth['ood'])
            ))

        # val as in-distribution, ood as out-of-distribution
        oracle = np.array([1]*len(truth['dev']) + [0]*len(truth['ood']))
        # oracle = np.array([1]*len(truth['test1']) + [0]*len(truth['ood']))
        # print("in_data {} ood_data {}".format(len(truth['dev']), len(truth['ood'])))

        for metric in uncertainty_res:
            metric_res = torch.load(os.path.join(src_dir, metric))
            metric_name = metric[:-4] # get rid of endswith '.res'
            ood_res[metric_name] = {}

            # average uncertainty
            self.cal_mUncertainty(metric_name, metric_res, ood_res)

            if metric_name not in ['Mutation', 'PVScore']:
                pred = np.concatenate((metric_res['dev'], metric_res['ood']))
                # pred = np.concatenate((metric_res['test1'], metric_res['ood']))
                AUC = common_get_auc(oracle, pred) # AUC
                AUPR = common_get_aupr(oracle, pred) # AUPR
                Brier = common_get_brier(oracle, pred) # Brier score
                print('AUC: %.4f, AUPR: %.4f, Brier: %.4f' % (AUC, AUPR, Brier))
                ood_res[metric_name] = {'AUC': AUC, 'AUPR': AUPR, 'Brier': Brier}
            else:
                preds = [
                    np.concatenate((val_res, ood_res))
                    # for val_res, ood_res in zip(metric_res['test1'], metric_res['ood'])
                    for val_res, ood_res in zip(metric_res['dev'], metric_res['ood'])
                ]
                for i, pred in enumerate(preds):
                    AUC = common_get_auc(oracle, pred) # AUC
                    AUPR = common_get_aupr(oracle, pred) # AUPR
                    Brier = common_get_brier(oracle, pred) # Brier score
                    print('(method %d) AUC: %.4f, AUPR: %.4f, Brier: %.4f' % (i+1, AUC, AUPR, Brier))

                ood_res[metric_name] = [
                    {
                        'AUC': common_get_auc(oracle, pred), 
                        'AUPR': common_get_aupr(oracle, pred), 
                        'Brier': common_get_brier(oracle, pred),
                    } for pred in preds
                ]
      
        # save evaluation res
        if not os.path.exists(os.path.join(self.save_dir, self.task)):
            os.makedirs(os.path.join(self.save_dir, self.task))
        save_name = os.path.join(self.save_dir, self.task, 'uncertainty_ood_eval.res')
        torch.save(ood_res, save_name)



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift_type', '-s', type=str, default='different_time', 
                        choices=[
                            'different_project', 
                            'different_author', 
                            'different_time',
                            'case_study',
                        ],
                        help='Type of code data shift.')
    parser.add_argument('--model', '-m', type=str, default='code2vec', help='Model name.')
    parser.add_argument('--task', '-t', type=str, default='code_summary', 
                        choices=['code_summary', 'code_completion'], help='Task name.')
    args = parser.parse_args()

    SHIFT = args.shift_type # different_project, different_author, different_time
    MODEL = args.model # code2vec, coderoberta, graphcodebert, lstm, codebert, codegpt
    TASK = 'CodeSummary_Module' if args.task == 'code_summary' else 'CodeCompletion_Module'

    eval_m = Uncertainty_Eval(
        res_dir='Uncertainty_Results/{}/{}'.format(SHIFT, MODEL),
        save_dir='Uncertainty_Eval/{}/{}'.format(SHIFT, MODEL), 
        task=TASK,
        shift=True if SHIFT != 'case_study' else False,
        ood=False, # True if ood is evaluated in Eval_res
    )
    # error/success prediction
    eval_m.evaluation()
