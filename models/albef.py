'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)

        self.prediction = nn.Sequential(
            nn.Linear(vision_width, vision_width // 2),
            nn.Tanh(),
            nn.Linear(vision_width // 2, 1)
        )

    def forward(
        self, image, input_ids, attention_mask, labels=None
    ):
        text_output = self.text_encoder.bert(input_ids, attention_mask = attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state # B x Tt x C
        
        image_embeds = self.visual_encoder(image.type_as(text_embeds)) # B x Ti x C
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # B x Ti
        
        cross_output = self.text_encoder.bert(
            encoder_embeds=text_embeds, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
            return_dict=True, mode='fusion'
        )
        cross_states = cross_output.last_hidden_state # B x (Tt+Ti) x C

        logits = self.prediction(cross_states[:, 0]).squeeze(1) # B
        if labels is None:
            return torch.sigmoid(logits.float())

        loss = F.binary_cross_entropy_with_logits(logits, labels.type_as(logits))

        return loss 