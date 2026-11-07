import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt

from utils_Informer.masking import ProbMask, TriangularCausalMask

class AttentionLayer(nn.Module):
    def __init__(self,inner_attention, middlesize, n_heads):
        super(AttentionLayer,self).__init__()
        d_keys = middlesize // n_heads
        d_values = middlesize // n_heads
        self.attention = inner_attention
        # self.attention = ProbAttention(False, factor=5, attention_dropout=0.0, output_attention=False)

        self.query_projection = nn.Linear(middlesize, d_keys * n_heads)
        self.key_projection = nn.Linear(middlesize, d_keys * n_heads)
        self.value_projection = nn.Linear(middlesize, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, middlesize)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape  # B:256, L:96, _:256
        _, S, _ = keys.shape
        H = self.n_heads  # H:8

        # 分别用三个矩阵得到q、k、v向量
        queries = self.query_projection(queries).view(B, L, H, -1)  # 注意这里出来是4维 512要分到H=8个头中 故为 8*64, [256, 96, 8, 64]
        keys = self.key_projection(keys).view(B, S, H, -1)  # [256, 96, 8, 64]
        values = self.value_projection(values).view(B, S, H, -1)  # [256, 96, 8, 64]

        out, attn = self.attention(  # 跳到38行的ProbAttention
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1) # [256, 96, 512]

        return self.out_projection(out), attn
        # 经过attention后 再经过一个全连接层 维度由512到512
        # attn一般为None

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape  # 256,8,96,64
        _, _, L_Q, _ = Q.shape  # 256,8,96,64

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # K_expand:[256, 8, 96, 96, 64] 即复制了96遍

        # 先unsqueeze使K 变为 [256, 8, 1, 96, 64]
        # expand()函数用于将张量沿指定的维度进行复制，并扩展到与目标张量相同的形状。
        # 在这里，expand(B, H, L_Q, L_K, E) 表示将 K_expand 张量沿 B、H、L_Q、L_K 和 E 维度进行复制和扩展，
        # 使得 K_expand 的形状与目标张量的形状相同，即 (B, H, L_Q, L_K, E)。

        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        # 产生一个[L_Q, sample_k]即[96, 25]的数组 元素为0~96的随机数
        # 为什么要对k抽25个呢？因为要对q进行抽样 现在有96个q 每个q都会对应96个k 但是每对应的这96个k我们不全看 只看25个 这样来看哪个q重要
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # K_sample是[batch_size, multi_heads, q中96个, k中取index_sample, 64]
        # K_sample:[256, 8, 96, 25, 64]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # Q_K_sample:[256, 8, 96, 25] 即96个q和25个k

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # 算96个q中与25个k关系最大的值 再计算与均匀分布的差异 即行最大值与均匀分布相比
        # M:[256, 8, 96]
        M_top = M.topk(n_top, sorted=False)[1]
        # 选25个最大的 M_top:[256, 8, 25]
        # [1]的意思是返回索引 [0]是返回数值

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        # 从96个找出那25个,Q_reduce:[256, 8, 25, 64]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        # Q_K:[256, 8, 25, 96]

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        # V:[256, 8, 96, 64]
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            # 沿倒数第二个维度求平均向量
            # V_sum:[256, 8, 64]
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
            # contex:[256, 8, 96, 64]而且向量全是一样的
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            # print('scores:', scores)
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        # 对25个q执行正常的softmax操作

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        # 按照索引值 将对应的25个q进行更新 其余没变的依旧是平均向量
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)
        # context_in:[256, 8, 96, 64]

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape  # 256, 96, 8, 64
        _, L_K, _, _ = keys.shape  # 96

        queries = queries.transpose(2, 1)  # [256, 8, 96, 64]
        keys = keys.transpose(2, 1)  # [256, 8, 96, 64]
        values = values.transpose(2, 1)  # [256, 8, 96, 64]

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # 0.5*ln(96)
        # =25，c*ln(L_k) 96个选了25个 U_part = 25
        # factor为5,log(L_K)为4.56,np.ceil是进行向上取整,得到5 相乘得25 然后由浮点转为int类型  .item() 将结果转换为一个标量值，即从数组中提取单个值。
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)  u = 25 对query进行同样的操作

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # scores_top:[256, 8, 25, 96] 25个q和96个k的attention
        # index是索引 [256, 8, 25]

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        # values:[256, 8, 96, 64],L_Q:96
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn

class FullAttention(nn.Module):  # cross attention和普通attention是完全一样的
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape # 128, 6, 8, 4
        _, S, _, D = values.shape # 128, 8, 8, 4
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # 即两个向量相乘求和 第一个向量维度为b、l、h、e，第二个为...，相乘得到...的结果
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)  # 将 scores 中对应于 attn_mask 中 True 位置的值用 -np.inf 进行填充。是一个原地操作，即地址操作

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 用0.5乘，即乘以dk，即softmax后的结果
        V = torch.einsum("bhls,bshd->blhd", A, values)  # 最后乘V

        if self.output_attention:  # 判断是否输出乘以V之前的结果
            return (V.contiguous(), A)  # 对张量 V 进行内存连续性处理，确保存储连续，提高计算效率
        else:
            return (V.contiguous(), None)
