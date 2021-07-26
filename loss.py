# from torch.nn import CrossEntropyLoss
from paddle.nn import CrossEntropyLoss
# from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
from paddle.nn.functional import loss
import paddle.nn as nn
class Loss(nn.Layer):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)
        # print('labels.dtype:', labels.dtype)
        # print('output.dtype:', outputs[0].dtype)
        CrossEntropy_Loss = [cross_entropy_loss(output, labels.astype('int64')) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum,
            Triplet_Loss,
            CrossEntropy_Loss),
              end=' ')
        return loss_sum
