import paddle
import paddle.nn as nn
import paddle.nn.functional as f

class FocalLoss(nn.Layer):
    """Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = f.log_softmax(input, axis=1)
        pt = paddle.exp(log_pt) # 回求prob true，正样本的概率值，因为上面已经取了对数
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = f.nll_loss( # 取负，求和
            log_pt,
            target,
            self.weight,
            reduction=self.reduction)
        return loss

def get_inv_freq(freq,plus=0,norm=False):
    inv_freq = [1 / f for f in freq]
    if norm:
        s=sum(inv_freq)
        inv_freq=[f/s for f in inv_freq]
    inv_freq=[f+plus for f in inv_freq]
    return inv_freq

if __name__ == '__main__':
    class_dim = 14
    logits=paddle.randn((4,class_dim))
    label=paddle.randint(0,14,(4,1))
    criterion=FocalLoss()
    loss=criterion(logits,label)
    print(loss)