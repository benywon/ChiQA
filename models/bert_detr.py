# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import DetrForObjectDetection
from models.xbert import BertConfig, BertForMaskedLM


class BertDetrQPicModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        bert_config = BertConfig.from_json_file(args.bert_config_file)
        self.text_encoder = BertForMaskedLM.from_pretrained(args.text_encoder_model, config=bert_config)
        self.image_encoder = DetrForObjectDetection.from_pretrained(args.image_encoder_model)
        self.hidden_dim = args.hidden_dim
        self.image_proj = nn.Linear(256, self.hidden_dim)
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

        image_output = self.image_encoder(image.type_as(text_embeds)) 
        image_embeds = self.image_proj(image_output.last_hidden_state) # B x 100 x C
        _, image_obj_ids = torch.max(image_output.logits, dim=2) # B x 100
        image_atts = image_obj_ids.ne(91).long() # B x 100
        image_obj_num = image_atts.sum(dim=1) # B
        # print(image_obj_num)
        image_zero_obj_ids = image_obj_num.eq(0).type_as(image_atts) # B 
        add_image_atts = image_atts.new_zeros(image_atts.size()) # B x 100
        add_image_atts[:, 0] = image_zero_obj_ids 
        image_atts = image_atts + add_image_atts # B x 100

        fusion_embeds = torch.cat([text_embeds, image_embeds], dim=1) # B x (T+100) x C
        fusion_atts = torch.cat([attention_mask, image_atts], dim=1) # B x (Tt + Ti)

        output = self.text_encoder.bert(
            encoder_embeds=fusion_embeds, # B x T x C
            attention_mask=fusion_atts,
            return_dict=True,
            mode='fusion',
        ) # B x T x C
        logits = output.last_hidden_state[:,0,:] # B x C
        logits = self.prediction(logits).squeeze(1) # B 
        if labels is None:
            return torch.sigmoid(logits.float())
        loss = F.binary_cross_entropy_with_logits(logits, labels.type_as(logits))

        return loss 