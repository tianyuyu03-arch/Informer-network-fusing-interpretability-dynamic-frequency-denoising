import torch

class TriangularCausalMask(): # 这样的掩码通常在序列到序列的模型中使用，以确保每个位置只能依赖于其之前的位置，防止信息泄漏。
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L] # 因果掩码的形状 第二维为通道数
        with torch.no_grad(): # 上下文管理器用于禁用梯度计算，因为构造掩码时不需要梯度信息
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            # triu函数创建一个上三角矩阵，上三角元素为1，其余为0
            # ones和triu组合，创建了一个布尔类型的张量，对角线以上为True，以下为False
    @property
    def mask(self):
        return self._mask # self._mask为私有属性，构造了三角因果掩码
    # 该方法提供了对_mask私有属性的只读访问，获取生成的三角因果掩码

class ProbMask(): # 生成概率掩码，这样的掩码通常在注意力机制中使用，用于屏蔽未来位置的信息，以确保在进行预测时只利用已知的信息。
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1) # 布尔类型的上三角掩码 triu(1)意为只保留对角线以上的元素
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        # 通过None在维度上增加两个维度，使_mask从原本形状[L, scores.shape[-1]]扩展为[1, 1, L, scores.shape[-1]]
        # .expend(B, H, L, scores.shape[-1])对_mask[1, 1, L, scores.shape[-1]]进行扩展
        # 使其在第一个维度复制B次， 在第二个维度复制H次，最终得到维度为(B, H, L, scores.shape[-1])
        # 这样_mask_ex就变成了一个与scores具有相同形状的概率掩码
        indicator = _mask_ex[torch.arange(B)[:, None, None],  # 先按照等差创建0~B-1到一维张量， 再增加二三维度为1， 维度为[B, 1, 1]
                             torch.arange(H)[None, :, None],  # [1, H, 1]
                             index, :].to(device)  # (B, H, L, scores.shape[-1])
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask