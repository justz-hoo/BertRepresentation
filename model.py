from transformers import BertTokenizer, BertConfig, BertModel
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torchdata


class TextModel(nn.Module):
    def __init__(self, config_path='bert-base-chinese/config.json',
                 model_path='bert-base-chinese/pytorch_model.bin'):
        super(TextModel, self).__init__()
        model_config = BertConfig.from_pretrained(config_path)
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        self.textExtractor = BertModel.from_pretrained(model_path, config=model_config)
        embedding_dim = self.textExtractor.config.hidden_size
        # self.fc = nn.Linear(in_features=embedding_dim, out_features=length)
        # self.tanh = nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        outputs = self.textExtractor(tokens,
                                     token_type_ids=segments,
                                     attention_mask=input_masks)
        text_embeddings = outputs[0][:, 0, :]
        # outputs[0] [batch size, sequence length, hidden_dimension]
        # features = self.fc(text_embeddings)
        # features = self.tanh(features)
        return text_embeddings
