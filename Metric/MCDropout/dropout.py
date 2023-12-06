import torch
from BasicalClass import BasicModule
from torch.nn import functional as F
from scipy.stats import entropy
from BasicalClass import common_predict, common_ten2numpy, common_get_maxpos
import numpy as np
from Metric import BasicUncertainty
from tqdm import tqdm

class ModelActivateDropout(BasicUncertainty):
    def __init__(self, instance: BasicModule, device, iter_time):
        super(ModelActivateDropout, self).__init__(instance, device)
        self.iter_time = iter_time

    def extract_metric(self, data_loader, orig_pred_y):
        res = 0
        self.model.train()
        for _ in range(self.iter_time):
            _, pred, _ = common_predict(
                data_loader, self.model, self.device, 
                module_id=self.module_id
            )
            res = res + pred.eq(orig_pred_y)
        self.model.eval()
        res = common_ten2numpy(res.float() / self.iter_time)
        return res

    def _predict_result(self, data_loader, model):
        # print('predicting result ...')
        pred_pos, pred_list, y_list = [], [], []
        model.to(self.device)

        if self.module_id == 0: # code summary
            for i, ((sts, paths, eds), y, length) in enumerate(data_loader):
                torch.cuda.empty_cache()
                sts = sts.to(self.device)
                paths = paths.to(self.device)
                eds = eds.to(self.device)
                y = torch.tensor(y, dtype=torch.long)
                output = model(starts=sts, paths=paths, ends=eds, length=length)
                _, pred_y = torch.max(output, dim=1) # shape: N
                # detach
                sts = sts.detach().cpu()
                paths = paths.detach().cpu()
                eds = eds.detach().cpu()
                pred_y = pred_y.detach().cpu()
                output = output.detach().cpu()

                pred_list.append(pred_y)
                pred_pos.append(output)
                y_list.append(y)

        elif self.module_id == 1: # code completion
            for i, (input, y, _) in enumerate(data_loader):
                torch.cuda.empty_cache()
                input = input.to(self.device)
                output = model(input)
                _, pred_y = torch.max(output, dim=1)
                # detach
                input = input.detach().cpu()
                pred_y = pred_y.detach().cpu()
                output = output.detach().cpu()
            
                # measure accuracy and record loss
                pred_list.append(pred_y)
                pred_pos.append(output)
                y_list.append(y.long())

        else:
            raise TypeError()
        return torch.cat(pred_pos, dim=0), torch.cat(pred_list, dim=0), torch.cat(y_list, dim=0)

    @staticmethod
    def label_chgrate(orig_pred, prediction):
        """
        orig_pred: N
        prediction: N X iter_time
        """
        _, repeat_num = np.shape(prediction) # N, iter_time
        tmp = np.tile(orig_pred.reshape([-1, 1]), (1, repeat_num)) # N X iter_time
        return np.sum(tmp == prediction, axis=1, dtype=np.float) / repeat_num
    
    def _pv(self, logits: torch.Tensor): # iter_time X N X k
        prob_scores = F.softmax(logits, dim=-1).numpy() # iter_time X N X k
        pv_scores = np.var(prob_scores, axis=0).mean(axis=1) # N
        return pv_scores
    
    def _bald(self, logits: torch.Tensor): # iter_time X N X k
        prob_scores = F.softmax(logits, dim=-1).numpy() # iter_time X N X k
        prob_mean = prob_scores.mean(axis=0) # \bar{p_k}: N X k
        mean_prob_uncertainties = entropy(prob_mean, axis=-1) # -sum p_k log p_k: N
        mean_sample_entropy = - entropy(prob_scores, axis=-1).mean(axis=0) # mean_t sum_k (p_k log p_k): N 
        return mean_sample_entropy + mean_prob_uncertainties # N

    def _uncertainty_calculate(self, data_loader):
        self.model.eval()
        smp_result, score_result = [], []
        print('Stochastic variational inference ...')
        self.model.train()
        for i in tqdm(range(self.iter_time)):
            score, _, _ = self._predict_result(data_loader, self.model) # N X k, N
            smp_result.append(common_get_maxpos(score).reshape([-1, 1])) # N X 1
            # score_result.append(score.detach().cpu()) # N X k
        print("Getting sampled maximum softmax response ...")
        smp_score = np.concatenate(smp_result, axis=1).mean(axis=1) # SMP: N
        print("SMP: ", smp_score.shape)
        return smp_score
        # print("Getting stacked score ...")
        # score_result = torch.stack(score_result, dim=0) # iter_time X N X k
        # print("Score: ", score_result.shape)
        # print("Calculating probability variance ...")
        # pv_score = - self._pv(score_result) # PV: N
        # print("PV: ", pv_score.shape)
        # print("Calculating BALD ...")
        # bald_score = - self._bald(score_result) # BALD: N
        # print("BALD: ", bald_score.shape)
        # return [smp_score, pv_score, bald_score]
