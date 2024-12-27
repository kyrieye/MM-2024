import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from models.transformer.Modules import ScaledDotProductAttention
class SemanticAlignment(nn.Module):
    def __init__(self, query_size, feat_size, bottleneck_size):
        super(SemanticAlignment, self).__init__()
        self.query_size = query_size
        self.feat_size = feat_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.query_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.feat_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)

    def forward(self, phr_feats, vis_feats):
        Wh = self.W(phr_feats)
        Uv = self.U(vis_feats)

        energies = self.w(torch.tanh(Wh[:, :, None, :] + Uv[:, None, :, :] + self.b)).squeeze(-1)
        weights = torch.softmax(energies, dim=2)
        aligned_vis_feats = torch.bmm(weights, vis_feats)
        return aligned_vis_feats, weights, energies

class EmotionAttention(nn.Module):
    def __init__(self, query_size, key_size, bottleneck_size):
        super(EmotionAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.query_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.key_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)

    def forward(self, query, keys, values, masks=None):
        topK= 20
        Wh = self.W(query)
        Uv = self.U(keys)
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.w(torch.tanh(Wh + Uv + self.b))
        if masks is not None:
            masks = masks[:, :, None]
            energies[masks] = -float('inf')
        weights = torch.softmax(energies, dim=1)

        dd_weights = weights.squeeze()
        top_wei, index = dd_weights.topk(topK, dim=1) 
        res = torch.zeros(dd_weights.size()).cuda()
        res = res.scatter(dim=1, index=index, src=top_wei) 
        res = res.unsqueeze(2).detach()

        weighted_feats = values * res.expand_as(values)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats, weights, energies

class SemanticAttention(nn.Module):
    def __init__(self, query_size, key_size, bottleneck_size):
        super(SemanticAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.query_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.key_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)

    def forward(self, query, keys, values, masks=None):
        Wh = self.W(query)
        Uv = self.U(keys)
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.w(torch.tanh(Wh + Uv + self.b))
        if masks is not None:
            masks = masks[:, :, None]
            energies[masks] = -float('inf')
        weights = torch.softmax(energies, dim=1)
        weighted_feats = values * weights.expand_as(values)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats, weights, energies

class Emcl(object):
    def __init__(self, k=32, stage_num=9, momentum=0.9, lamd=1, beta=3):
        self.k = k
        self.lamd = lamd
        self.stage_num = stage_num
        self.beta = beta
        self.momentum = momentum
        self.mu = torch.Tensor(1, self.k)
        self.mu.normal_(0, math.sqrt(2. / self.k))
        self.mu = self.mu / (1e-6 + self.mu.norm(dim=0, keepdim=True))

    def __call__(self, embds, if_train=True):
        b, n = embds.size()
        mu = self.mu.repeat(b, 1).cuda(embds.device)
        _embds = embds
        with torch.no_grad():
            for i in range(self.stage_num):
                _embds_t = _embds.permute(1, 0)  # n * b
                z = torch.mm(_embds_t, mu)  # n * k
                z = z / self.lamd
                z = F.softmax(z, dim=1)
                z = z / (1e-6 + z.sum(dim=0, keepdim=True))
                mu = torch.mm(_embds, z)  # b * k
                mu = mu / (1e-6 + mu.norm(dim=0, keepdim=True))
        z_t = z.permute(1, 0)  # k * n
        _embds = torch.mm(mu, z_t)  # b * n

        if if_train:
            mu = mu.cpu()
            self.mu = self.momentum * self.mu + (1 - self.momentum) * mu.mean(dim=0, keepdim=True)
        return self.beta * _embds + embds

class DynamicSemanticModule(nn.Module):
    def __init__(self,emo_size,feat_size,k,h_list) -> None:
        super(DynamicSemanticModule,self).__init__()
        self.emo_size = emo_size
        self.feat_size = feat_size
        self.aggregation_k = k
        self.h_list = h_list
        self.d_list = []
        for i in self.h_list:
            self.d_list.append(int(self.emo_size/i))
        self.d_sum = int(sum(self.d_list))
        self.h_sum = int(sum(self.h_list))
        self.alpha = 1

        self.agg_pool = nn.AdaptiveAvgPool1d
        self.w_v = nn.Linear(self.feat_size,self.feat_size,bias=True)
        self.agg_w_a = nn.Linear(self.feat_size,self.aggregation_k,bias=True)
        self.agg_w_c = nn.Linear(self.feat_size,self.emo_size)
        self.agg_softmax = nn.Softmax(dim=-1)

        self.ele_route_w = nn.Linear(2 * self.emo_size, 1)
        self.ele_route_tanh = nn.Tanh()

        self.subspace_wq = nn.Linear(self.emo_size,self.d_sum,bias=True)
        self.subspace_wk = nn.Linear(self.emo_size,self.d_sum,bias=True)
        self.subspace_wv = nn.Linear(self.emo_size,self.h_sum,bias=True)
        self.subspace_wr = nn.Linear(self.emo_size,len(self.h_list),bias=True)
        self.output_atten = ScaledDotProductAttention(temperature=np.power(self.d_sum, 0.5))
        self.output_tanh = nn.Tanh()
        self.R_atten = ScaledDotProductAttention(temperature=np.power(self.d_sum, 0.5))
        self.R_softmax = nn.Softmax()
        

    def forward(self,emo_embedding,phr_feat,video_feat):
        aggre_feat = self.feature_aggregation(phr_feat,video_feat)
        route_emo_feat = self.element_routing(aggre_feat,emo_embedding)
        new_emo_feat = self.subspace_routing(route_emo_feat,aggre_feat,emo_embedding)
        return new_emo_feat

    def feature_aggregation(self,phr_feat,video_feat):
        N = video_feat.shape[1]
        phr_feat = self.agg_pool(N)(phr_feat.transpose(1,2)).transpose(1,2)
        combine_feat = torch.concat([self.w_v(phr_feat),video_feat],dim=1)
        A_agg = self.agg_w_a(combine_feat)
        feat_a = self.agg_w_c(combine_feat)
        feat = torch.bmm((self.agg_softmax(A_agg)).transpose(1,2),feat_a)
        return feat
    
    def element_routing(self,agg_feat,emo_embedding):
        A_cross = torch.bmm(emo_embedding,agg_feat.transpose(1,2))
        A_cross_pool = torch.mean(A_cross,dim=1).reshape(-1,1,self.aggregation_k)
        route_feat = torch.bmm(A_cross_pool,agg_feat).repeat(1,emo_embedding.shape[1],1)
        concat_feat = torch.concat([emo_embedding,route_feat],dim=-1)
        R_e = self.alpha * self.ele_route_tanh(self.ele_route_w(concat_feat))
        route_emo_feat = R_e * emo_embedding
        return route_emo_feat

    def subspace_routing(self,route_emo_feat,aggre_feat,emo_embedding):
        B = route_emo_feat.shape[0]
        L_t = route_emo_feat.shape[1]
        #M_Q
        M_Q_list = []
        for i in range(len(self.d_list)):
            M_Q = torch.zeros(size=(L_t,self.d_sum)) # [L_t,d_sum]
            if i==0:
                M_Q[:,:self.d_list[i]] = 1
                M_Q_list.append(M_Q)
            else:
                left = 0
                for j in range(i):
                    left += self.d_list[j]
                right = left+self.d_list[i]
                M_Q[:,left:right] = 1
                M_Q_list.append(M_Q)
        N = len(M_Q_list)
        M_Q = torch.concat(M_Q_list,dim=0).reshape(N,L_t,self.d_sum).repeat(B,1,1).reshape(B,N*L_t,self.d_sum).cuda() # [B,N*L_t,d_sum]
        #M_V
        M_V_list = []
        for i in range(len(self.h_list)):
            M_V = torch.zeros(size=(self.aggregation_k,self.h_sum)) # [K,h_sum]
            if i==0:
                M_V[:,:self.h_list[i]] = 1
                M_V_list.append(M_V)
            else:
                left = 0
                for j in range(i):
                    left += self.h_list[j]
                right = left+self.h_list[i]
                M_V[:,left:right] = 1
                M_V_list.append(M_V)
        M_V = torch.concat(M_V_list,dim=0).reshape(N,self.aggregation_k,self.h_sum).repeat(B,1,1).reshape(B,N*self.aggregation_k,self.h_sum).cuda() #[B,N*k,h_sum]
        
        Q = self.subspace_wq(route_emo_feat).repeat(1,N,1).reshape(B,N*L_t,self.d_sum) # [B,N*L_t,d_sum]
        K = self.subspace_wk(aggre_feat).repeat(1,N,1).reshape(B,N*self.aggregation_k,self.d_sum) # [B,N*k,d_sum]
        V = self.subspace_wv(aggre_feat).repeat(1,N,1).reshape(B,N*self.aggregation_k,self.h_sum) #[B,N*k,h_sum]
        R = self.subspace_wr(aggre_feat).repeat(1,N,1).reshape(B,N*self.aggregation_k,N) #[B,N*k,N]
        
        O = self.output_tanh(self.output_atten(torch.mul(Q,M_Q),K,torch.mul(V,M_V))[0]).reshape(B,N,L_t,self.h_sum)  #[B,N,L_t,h_sum]
        O_R = self.R_atten(torch.mul(Q,M_Q),K,R)[0].reshape(B,N,L_t,N) #[B,N,L_t,N]

        T_list = []
        for i in range(N):
            ori_T_i = route_emo_feat.reshape(B,L_t,self.h_list[i],self.d_list[i])  # [B,L_t,h_j,d_j]
            if i==0:
                O_i = O[:,i,:,:self.h_list[i]].reshape(B,L_t,self.h_list[i],1) # [B,L_t,h_j,1]
            else:
                left = 0
                for j in range(i):
                    left += self.h_list[j]
                right = left+self.h_list[i]
                O_i = O[:,i,:,left:right].reshape(B,L_t,self.h_list[i],1) # [B,L_t,h_j,1]
            new_T_i = torch.mul(ori_T_i,O_i) # [B,L_t,h_j,d_j]
            new_T_i = new_T_i.reshape(B,L_t,-1) #[B,L_T,D_t]
            T_list.append(new_T_i)
        T = torch.concat(T_list,dim=0).reshape(N,B,L_t,-1).transpose(0,1).cuda() #[B,N,L_t,D_t]

        R_list = []
        for j in range(N):
            R_i = torch.mean(O_R[:,j,:,j].reshape(B,L_t),dim=-1).reshape(1,B) #[1,B]
            R_list.append(R_i)
        R = torch.concat(R_list,dim=0).reshape(N,B).T.cuda() #[B,N]

        R_sqr = self.R_softmax(R).reshape(B,N,1) #[B,N]
        R_res = R_sqr.repeat(1,1,L_t).reshape(B,N,L_t,1)
        residual = torch.mul(R_res,T)
        new_emo_feat = emo_embedding + torch.sum(residual,dim=1) #[B,L_t,D_t]
        return new_emo_feat


class DynamicEmotionalShiftModule(nn.Module):
    def __init__(self,emo_size,feat_size,k,h_list) -> None:
        super(DynamicSemanticModule,self).__init__()
        self.emo_size = emo_size
        self.feat_size = feat_size
        self.aggregation_k = k
        self.h_list = h_list
        self.d_list = []
        for i in self.h_list:
            self.d_list.append(int(self.emo_size/i))
        self.d_sum = int(sum(self.d_list))
        self.h_sum = int(sum(self.h_list))
        self.alpha = 1

        self.agg_pool = nn.AdaptiveAvgPool1d
        self.w_v = nn.Linear(self.feat_size,self.feat_size,bias=True)
        self.agg_w_a = nn.Linear(self.feat_size,self.aggregation_k,bias=True)
        self.agg_w_c = nn.Linear(self.feat_size,self.emo_size)
        self.agg_softmax = nn.Softmax(dim=-1)

        self.ele_route_w = nn.Linear(2 * self.emo_size, 1)
        self.ele_route_tanh = nn.Tanh()

        

    def forward(self,emo_embedding,phr_feat,video_feat):
        aggre_feat = self.feature_aggregation(phr_feat,video_feat)
        route_emo_feat = self.element_routing(aggre_feat,emo_embedding)
        new_emo_feat = self.emotion_shift(route_emo_feat,aggre_feat,emo_embedding)
        return new_emo_feat

    def feature_aggregation(self,phr_feat,video_feat):
        N = video_feat.shape[1]
        phr_feat = self.agg_pool(N)(phr_feat.transpose(1,2)).transpose(1,2)
        combine_feat = torch.concat([self.w_v(phr_feat),video_feat],dim=1)
        A_agg = self.agg_w_a(combine_feat)
        feat_a = self.agg_w_c(combine_feat)
        feat = torch.bmm((self.agg_softmax(A_agg)).transpose(1,2),feat_a)
        return feat
    
    def element_routing(self,agg_feat,emo_embedding):
        A_cross = torch.bmm(emo_embedding,agg_feat.transpose(1,2))
        A_cross_pool = torch.mean(A_cross,dim=1).reshape(-1,1,self.aggregation_k)
        route_feat = torch.bmm(A_cross_pool,agg_feat).repeat(1,emo_embedding.shape[1],1)
        concat_feat = torch.concat([emo_embedding,route_feat],dim=-1)
        R_e = self.alpha * self.ele_route_tanh(self.ele_route_w(concat_feat))
        route_emo_feat = R_e * emo_embedding
        return route_emo_feat
    
    def emotion_shift(self):
        pass