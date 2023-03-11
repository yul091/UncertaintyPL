import torch
import torch.nn as nn
from transformers import (
    RobertaModel, 
    RobertaPreTrainedModel,
    GPT2Model,
    GPT2PreTrainedModel,
)
from models.utils import count_parameters


class Word2vecPredict(nn.Module):
    def __init__(self, vocab_size, token_vec, hidden_size=120):
        super(Word2vecPredict, self).__init__()
        if torch.is_tensor(token_vec):
            self.encoder = nn.Embedding(vocab_size, hidden_size, padding_idx=0, _weight=token_vec)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.linear = nn.Linear(hidden_size, vocab_size)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]

    def forward(self, x): # B X T
        vec = self.encoder(x) # B X T X H
        vec = torch.mean(vec, dim=1) # B X H
        pred = self.linear(vec) # B X V
        return pred

    def get_hidden(self, x):
        res = []
        vec = self.encoder(x)
        vec = torch.mean(vec, dim=1)
        res.append(vec.detach().cpu())
        # pred = self.linear(vec)
        return res


class BiLSTMForClassification(nn.Module):
    def __init__(self, vocab_size, token_vec, hidden_size=120, num_layers=2, dropout=0.1):
        super().__init__()
        if torch.is_tensor(token_vec):
            self.encoder = nn.Embedding(vocab_size, hidden_size, padding_idx=0, _weight=token_vec)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.activation = nn.Tanh()
        self.linear = nn.Linear(hidden_size, vocab_size)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]
        
    def forward(self, x):
        embed = self.encoder(x) # B X T X H
        _, hidden = self.lstm(embed) # 2*num_layers X B X H
        h, c = hidden
        last_hidden = h[-1] + c[-1] # B X H
        pooled_output = self.activation(last_hidden) # B X H
        pred = self.linear(pooled_output) # B X V
        
        return pred

    def get_hidden(self, x):
        res = []
        embed = self.encoder(x) # B X T X H
        _, hidden = self.lstm(embed) # 2*num_layers X B X H
        h, c = hidden
        last_hidden = h[-1] + c[-1] # B X H
        pooled_output = self.activation(last_hidden) # B X H
        res.append(pooled_output.detach().cpu())
        
        return res


class CodeBertForClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config, dropout=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sub_num = [1]
        self.init_weights()
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
            
    
    def forward(self, input_ids):
        outputs = self.roberta(input_ids)
        pooled_output = outputs['pooler_output'] # B X H
        pooled_output = self.dropout(pooled_output) # B X H
        pred = self.classifier(pooled_output) # B X V
        
        return pred

    def get_hidden(self, x):
        res = []
        outputs = self.roberta(x)
        pooled_output = outputs['pooler_output'] # B X H
        res.append(pooled_output.detach().cpu())
        pooled_output = self.dropout(pooled_output) # B X H

        return res
    
    
class CodeGPTForClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]
    
    def __init__(self, config, dropout: float = 0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.sub_num = [1]
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(self, input_ids: torch.LongTensor):
        outputs = self.gpt2(input_ids)
        hidden_states = outputs['last_hidden_state'] # B X T X H
        hidden_states = self.dropout(hidden_states) # B X T X H
        logits = self.classifier(hidden_states) # B X T X V
        batch_size, sequence_length = input_ids.shape[:2]
        
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits

    def get_hidden(self, x: torch.LongTensor):
        res = []
        outputs = self.gpt2(x)
        hidden_states = outputs['last_hidden_state'] # B X T X H
        res.append(hidden_states[:, -1, :].detach().cpu())
        hidden_states = self.dropout(hidden_states)
        return res