import paddle
import paddle.nn as nn


class TripletLoss(nn.Layer):
    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        # print('inputs:', inputs)
        n = inputs.shape[0]
        dist = paddle.pow(inputs, 2).sum(axis=1, keepdim=True).expand(shape=[n, n])

        dist = dist + dist.t()

        # dist = paddle.addmm(dist, inputs, inputs.t(), -2, 1)
        dist = paddle.addmm(input=dist, x=inputs, y=inputs.t(), beta=1, alpha=-2)
        # print('\n\n\n\n\ndist:', dist)
        # dist.addmm(inputs=-2,x=1,y=-2, alpha=inputs, beta=1,)

        dist = dist.clip(min=1e-12).sqrt()

        mask = targets.expand(shape=[n, n]).equal(targets.expand(shape=[n, n]).t())

        # dist = dist.numpy()
        # mask = mask.numpy()

        # mask = mask.astype('float32')
        # print(mask)
        # for i in range(n):
        #     for j in range(n):
        #         # print(mask[i][j].numpy())
        #         if mask[i][j].numpy().item():
        #             mask[i][j]=1
        #         else:
        #             mask[i][j]=0
        # print('mask:', mask[0])

        dist_ap, dist_an = [], []

        dist_ap_temp = paddle.masked_select(dist, mask)
        dist_an_temp = paddle.masked_select(dist, mask.logical_not())

        # print('mask:', mask.numpy()[0])
        # print('mask:', type(mask))
        for i in range(n):
            # print('dist:', dist)
            # print('mask:', mask[i])
            # for j in range(n):
            #     if mask[i] ==1
            # print('dist_ap:', dist_ap_temp)
            dist_ap.append(dist_ap_temp[i].max().unsqueeze(0))
            dist_an.append(dist_an_temp[i].min().unsqueeze(0))
        dist_ap = paddle.concat(dist_ap)
        dist_an = paddle.concat(dist_an)
        y = paddle.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
