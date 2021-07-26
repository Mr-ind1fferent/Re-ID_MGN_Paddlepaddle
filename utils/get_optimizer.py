# from torch.optim import Adam, SGD
from opt import opt
from paddle.optimizer import Adam
import paddle
###wait
def get_optimizer(net):
    scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=opt.lr, milestones=opt.lr_scheduler, gamma=0.1)

    if opt.freeze:

        for p in net.parameters():
            p.requires_grad = True
        for q in net.backbone.parameters():
            q.requires_grad = False

        optimizer = Adam(learning_rate=scheduler,parameters=filter(lambda p: p.requires_grad, net.parameters()),  weight_decay=5e-4,
                         )

    else:

        # optimizer = SGD(net.parameters(), lr=opt.lr,momentum=0.9, weight_decay=5e-4)
        optimizer = Adam(learning_rate=scheduler,parameters=net.parameters(), weight_decay=5e-4, )

    return optimizer
