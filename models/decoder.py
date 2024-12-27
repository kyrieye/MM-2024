import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from models.attention import SemanticAlignment, SemanticAttention, EmotionAttention,DynamicSemanticModule
from models.transformer.Models import Encoder_emo
from models.transformer.Models import Decoder_emo_nomul
class Decoder(nn.Module):
    def __init__(self, num_layers, emo_word_size ,vis_feat_size, feat_len, embedding_size, sem_align_hidden_size,
                 sem_attn_hidden_size, hidden_size, output_size,emo_atten):
        super(Decoder, self).__init__()
        self.emo_alpha=0.8
        self.emo_word_size = emo_word_size
        self.num_layers = num_layers
        self.vis_feat_size = vis_feat_size
        self.feat_len = feat_len
        self.embedding_size = embedding_size
        self.sem_align_hidden_size = sem_align_hidden_size
        self.sem_attn_hidden_size = sem_attn_hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.emo_atten=emo_atten
        #self.sc = Encoder_emo(n_layers=1, n_head=1, d_k=32, d_v=32, d_model=300, d_inner=512)
        #self.ca = Decoder_emo_nomul(n_layers=4,n_head=4,d_k=32,d_v=32,d_model=300,d_inner=300)
        self.semantic_attention_word = SemanticAttention(
            query_size=self.hidden_size,
            key_size=self.embedding_size,
            bottleneck_size=self.sem_attn_hidden_size)

        self.semantic_attention_vis = SemanticAttention(
            query_size=self.hidden_size,
            key_size=self.vis_feat_size,
            bottleneck_size=self.sem_attn_hidden_size)
        
        self.semantic_attention_vw = SemanticAttention(
            query_size=self.hidden_size,
            key_size=self.embedding_size,
            bottleneck_size=self.sem_attn_hidden_size)

        self.semantic_alignment = SemanticAlignment(
            query_size=self.vis_feat_size,
            feat_size=self.embedding_size,
            bottleneck_size=self.sem_align_hidden_size)

        self.semantic_attention_all= SemanticAttention(
            query_size = self.hidden_size,
            key_size = self.vis_feat_size * 2 + self.embedding_size,
            bottleneck_size = self.sem_attn_hidden_size
        )

        self.intensity_alignment = SemanticAlignment(
            query_size=self.vis_feat_size,
            feat_size=self.embedding_size,
            bottleneck_size=self.sem_align_hidden_size)
        self.dynamic_module = DynamicSemanticModule(self.embedding_size,self.embedding_size,100,[2,3,5,6,10])
        self.rnn = nn.LSTM(
            input_size=self.vis_feat_size * 2 + self.embedding_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def get_last_hidden(self, hidden):
        last_hidden = hidden[0]
        last_hidden = last_hidden.view(self.num_layers, 1, last_hidden.size(1), last_hidden.size(2))
        last_hidden = last_hidden.transpose(2, 1).contiguous()
        last_hidden = last_hidden.view(self.num_layers, last_hidden.size(1), last_hidden.size(3))
        last_hidden = last_hidden[-1]
        return last_hidden


    def sp(self,em_embdeding,vis_feats,last_hidden,hidden,embedded,semantic_group_feats,phr_feats):
        B = em_embdeding.shape[0]
        intensity_feat,_,_ = self.intensity_alignment(
            phr_feats=em_embdeding,
            vis_feats=phr_feats)
        
        emo_inten = torch.sigmoid(torch.mean(intensity_feat,dim=-1)).reshape(B,-1,1)
        
        feat_evw_c = torch.cat((
            emo_inten*em_embdeding,
            semantic_group_feats,
            vis_feats),dim=2)

        feat_evw, dec_weights, _ = self.semantic_attention_all(
            query=last_hidden,
            keys=feat_evw_c,
            values=feat_evw_c)

        embedded_word = embedded 

        feat = torch.cat((
            feat_evw,
            embedded_word), dim=1)
        
        
        output, hidden = self.rnn(feat[None, :, :], hidden)
        
        output = output.squeeze(0)
        output = self.out(output)
        output = torch.log_softmax(output, dim=1)
        return output,hidden

    def forward(self, sub_emo,em_embdeding, embedded, hidden, vis_feats, phr_feats, t,emotion_vocab_feats,emotion_cate_feats,sub_mode):

        last_hidden = self.get_last_hidden(hidden)
        
        semantic_group_feats, semantic_align_weights, semantic_align_logits = self.semantic_alignment(
            phr_feats=vis_feats,
            vis_feats=phr_feats)
        
        if sub_mode == True:
            objective_emo = self.dynamic_module(sub_emo,phr_feats,vis_feats)
            #em_embdeding, *_ = self.sc(em_embdeding)
            #em_embdeding,_,_ = self.ca(vis_feats,em_embdeding)
            mix_emo = self.emo_alpha*em_embdeding + (1-self.emo_alpha)*objective_emo
            output, hidden = self.sp(mix_emo,vis_feats,last_hidden,hidden,embedded,semantic_group_feats,phr_feats)
        else:
            em_embdeding = self.dynamic_module(em_embdeding,phr_feats,vis_feats)
            #em_embdeding, *_ = self.sc(em_embdeding)
            #em_embdeding,_,_ = self.ca(vis_feats,em_embdeding)
            output, hidden = self.sp(em_embdeding,vis_feats,last_hidden,hidden,embedded,semantic_group_feats,phr_feats)
        return output, hidden, semantic_align_weights, semantic_align_logits,em_embdeding

