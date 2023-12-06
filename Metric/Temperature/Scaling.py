import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from BasicalClass import (
    common_predict, 
    common_get_maxpos, 
    common_cal_accuracy, 
    common_ten2numpy,
    spearmanr,
)
from BasicalClass import BasicModule
from Metric import BasicUncertainty


class ModelWithTemperature(BasicUncertainty):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, instance: BasicModule, device, temperature=None):
        super(ModelWithTemperature, self).__init__(instance, device)
        if temperature is None:
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            self.set_temperature(self.val_loader)
        else:
            self.temperature = temperature
        torch.save(self.temperature, os.path.join(
            instance.save_dir, instance.__class__.__name__, 'temperature.tmp'
        ))

    # modified forward for code summary task
    def forward(self, *input, **kwargs):
        # here since the model is code_summary model, the input has to be changed
        logits = self.model(*input, **kwargs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # First: collect all the logits and labels for the validation set
        # logits_list = []
        # labels_list = []
        # with torch.no_grad():
        #     if self.module_id == 0: # code summary
        #         for i, ((sts, paths, eds), y, length) in enumerate(valid_loader):
        #             torch.cuda.empty_cache()
        #             sts = sts.to(self.device)
        #             paths = paths.to(self.device)
        #             eds = eds.to(self.device)
        #             y = torch.tensor(y, dtype=torch.long)
        #             logits = self.model(starts=sts, paths=paths, ends=eds, length=length)
                    
        #             # detach
        #             sts = sts.detach().cpu()
        #             paths = paths.detach().cpu()
        #             eds = eds.detach().cpu()

        #             if isinstance(logits, tuple):
        #                 logits = (py.detach().cpu() for py in logits)
        #             else:
        #                 logits = logits.detach().cpu()

        #             logits_list.append(logits)
        #             labels_list.append(y)

        #     elif self.module_id == 1: # code completion
        #         for i, (input, y, _) in enumerate(valid_loader):
        #             torch.cuda.empty_cache()
        #             input = input.to(self.device)
        #             logits = self.model(input) # shape: N X class_num
        #             logits_list.append(logits.detach().cpu())
        #             labels_list.append(y.long())
                    
        #     else:
        #         raise TypeError()

        #     logits = torch.cat(logits_list) # shape: N X class_num
        #     labels = torch.cat(labels_list) # shape: N
        logits, _, labels = common_predict(valid_loader, self.model, self.device, module_id=self.module_id)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = self.nll_criterion(logits, labels).item()
        before_temperature_ece = self.ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = self.nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = self.nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = self.ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        return self

    def _uncertainty_calculate(self, data_loader):
        logits, preds, labels = common_predict(data_loader, self, self.device, module_id=self.module_id)
        uncertainty = common_get_maxpos(logits) # winning score as uncertainty
        nll = self.nll_criterion(self.temperature_scale(logits), labels).item()
        ece = self.ece_criterion(self.temperature_scale(logits), labels).item()
        acc = common_cal_accuracy(preds, labels)
        rank_correlation, _ = spearmanr(uncertainty, preds.eq(labels).float())
        print('Acc: %.4f, NLL: %.4f, ECE: %.4f,  Spearman rank correlation: %.4f' % (acc, nll, ece, rank_correlation))
        return {
            'UE_scores': uncertainty,
            'preds': common_ten2numpy(preds),
            'labels': common_ten2numpy(labels),
            'nll': nll,
            'ece': ece,
            'acc': acc,
            'rank_correlation': rank_correlation
        }

