import os

import torch
#from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_xlnet import XLNetConfig,XLNetModel
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_xlnet import XLNetTokenizer 
from torch import nn


class BertModule(nn.Module):
    def __init__(self, bert_model_name, cache_dir="./cache/"):
        super().__init__()

        # Create cache directory if not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.bert_model = XLNetModel.from_pretrained(
            bert_model_name, cache_dir=cache_dir
        )

    def forward(self, token_ids, token_type_ids=None, attention_mask=None):
        encoded_layers, pooled_output = self.bert_model(
            token_ids, token_type_ids, attention_mask
        )
        return encoded_layers, pooled_output


class BertLastCLSModule(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input):
        #file1 = open("/content/xlnn/superglue_modules/debug.txt","a") 
        #print((input.shape)[0])
        last_hidden = input[-1][0]
    
        out = self.dropout(last_hidden)
        return out


class BertContactLastCLSWithTwoTokensModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, idx1, idx2):
        last_layer = input[-1]
        last_cls = last_layer[ 0, :]
        idx1 = idx1.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, last_layer.size(-1)])
        idx2 = idx2.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, last_layer.size(-1)])
        token1_emb = last_layer.gather(dim=1, index=idx1).squeeze(dim=1)
        token2_emb = last_layer.gather(dim=1, index=idx2).squeeze(dim=1)
        output = torch.cat([last_cls, token1_emb, token2_emb], dim=-1)
        return output
