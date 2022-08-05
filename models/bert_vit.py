# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import AutoModel
from models.xbert import BertConfig, BertForMaskedLM
import random 


class BertViTQPicModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        bert_config = BertConfig.from_json_file(args.bert_config_file)
        self.text_encoder = BertForMaskedLM.from_pretrained(args.text_encoder_model, config=bert_config)
        self.vit_encoder = AutoModel.from_pretrained(args.vit_encoder_model)
        self.hidden_dim = args.hidden_dim
        self.prediction = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def forward(
        self, input_ids, attention_mask, 
        image, labels=None
    ):
        text_output = self.text_encoder.bert(
            input_ids, attention_mask=attention_mask,
            return_dict=True, mode='text'
        ) 
        text_embeds = text_output.last_hidden_state # B x T x C
        image_output = self.vit_encoder(image.type_as(text_embeds)) 
        image_embeds = image_output.last_hidden_state # B x T x C
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # B x T
        fusion_embeds = torch.cat([text_embeds, image_embeds], dim=1) # B x (Tt+Ti) x C
        fusion_atts = torch.cat([attention_mask, image_atts], dim=1) # B x (Tt + Ti)
        
        output = self.text_encoder.bert(
            encoder_embeds=fusion_embeds, # B x T x C
            attention_mask=fusion_atts,
            return_dict=True,
            mode='fusion',
        ) # B x T x C
        logits = output.last_hidden_state[:,0,:] # B x C
        logits = self.prediction(logits).squeeze(1) # B x 1 -> B 
        if labels is None:
            return torch.sigmoid(logits.float())
        
        loss = F.binary_cross_entropy_with_logits(logits, labels.type_as(logits))

        return loss 
    
    def get_ita(self, image, text):
        text_output = self.text_encoder.bert(
            text.input_ids, attention_mask=text.attention_mask,
            return_dict=True, mode='text'
        ) 
        text_embeds = text_output.last_hidden_state # B x T x C 
        text_feat = F.normalize(text_embeds[:, 0, :], dim=-1) # B x C
        image_output = self.vit_encoder(image) 
        image_embeds = image_output.last_hidden_state # B x T x C
        image_feat = F.normalize(image_embeds[:, 0, :]) # B x C
        logits = (text_feat * image_feat).sum(dim=-1) # B
        return logits 