import paddle
import copy
from paddle.vision.models.resnet import resnet50, BottleneckBlock
import paddle.nn as nn
from paddle.nn.initializer.kaiming import KaimingNormal

num_classes = 751  # change this depend on your dataset


class MGN(nn.Layer):

    def __init__(self):
        super(MGN, self).__init__()
        feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )
        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        # res_conv5 global
        res_g_conv5 = resnet.layer4
        # res_conv5 part
        res_p_conv5 = nn.Sequential(
            BottleneckBlock(1024, 512, downsample=nn.Sequential(
                nn.Conv2D(1024, 2048, 1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal(fan_in=True)),
                nn.BatchNorm2D(2048))),
            BottleneckBlock(2048, 512),
            BottleneckBlock(2048, 512))

        res_p_conv5.state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2D(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2D(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2D(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2D(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2D(kernel_size=(8, 8))

        bn_weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Normal(mean=1.0, std=0.02))

        self.reduction = nn.Sequential(
            nn.Conv2D(2048, feats, 1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm2D(feats, weight_attr=bn_weight_attr),
            nn.ReLU())


        # fc softmax loss


        self.fc_id_2048_0 = nn.Linear(feats,
                                      num_classes,
                                      weight_attr=nn.initializer.KaimingNormal(),
                                      bias_attr=nn.initializer.Constant(value=0.0))
        self.fc_id_2048_1 = nn.Linear(feats,
                                      num_classes,
                                      weight_attr=nn.initializer.KaimingNormal(),
                                      bias_attr=nn.initializer.Constant(value=0.0))
        self.fc_id_2048_2 = nn.Linear(feats,
                                      num_classes,
                                      weight_attr=nn.initializer.KaimingNormal(),
                                      bias_attr=nn.initializer.Constant(value=0.0))
        self.fc_id_256_1_0 = nn.Linear(feats,
                                       num_classes,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=nn.initializer.Constant(value=0.0))
        self.fc_id_256_1_1 = nn.Linear(feats,
                                       num_classes,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=nn.initializer.Constant(value=0.0))
        self.fc_id_256_2_0 = nn.Linear(feats,
                                       num_classes,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=nn.initializer.Constant(value=0.0))
        self.fc_id_256_2_1 = nn.Linear(feats,
                                       num_classes,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=nn.initializer.Constant(value=0.0))
        self.fc_id_256_2_2 = nn.Linear(feats,
                                       num_classes,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=nn.initializer.Constant(value=0.0))

    def forward(self, x):
        # print('inputs.shape:', x.shape)
        x = self.backbone(x)
        # print('x.shape:', x.shape)
        p1 = self.p1(x)
        # print('p1.shape:', p1.shape)
        p2 = self.p2(x)
        # print('p2.shape:', p2.shape)
        p3 = self.p3(x)
        # print('p3.shape:', p3.shape)

        zg_p1 = self.maxpool_zg_p1(p1)
        # print('zg_p1.shape:', zg_p1.shape)
        zg_p2 = self.maxpool_zg_p2(p2)
        # print('zg_p2.shape:', zg_p2.shape)
        zg_p3 = self.maxpool_zg_p3(p3)
        # print('zg_p3.shape:', zg_p3.shape)

        zp2 = self.maxpool_zp2(p2)
        # print('zp2.shape:', zp2.shape)
        z0_p2 = zp2[:, :, 0:1, :]
        # print('z0_p2.shape:', z0_p2.shape)
        z1_p2 = zp2[:, :, 1:2, :]
        # print('z1_p2.shape:', z1_p2.shape)

        zp3 = self.maxpool_zp3(p3)
        # print('zp3.shape:', zp3.shape)
        z0_p3 = zp3[:, :, 0:1, :]
        # print('z0_p3.shape:', z0_p3.shape)
        z1_p3 = zp3[:, :, 1:2, :]
        # print('z1_p3.shape:', z1_p3.shape)
        z2_p3 = zp3[:, :, 2:3, :]
        # print('z2_p3.shape:', z2_p3.shape)

        fg_p1 = self.reduction(zg_p1).squeeze(axis=3).squeeze(axis=2)
        # print('fg_p1.shape:', fg_p1.shape)
        fg_p2 = self.reduction(zg_p2).squeeze(axis=3).squeeze(axis=2)
        # print('fg_p2.shape:', fg_p2.shape)
        fg_p3 = self.reduction(zg_p3).squeeze(axis=3).squeeze(axis=2)
        # print('fg_p3.shape:', fg_p3.shape)
        f0_p2 = self.reduction(z0_p2).squeeze(axis=3).squeeze(axis=2)
        # print('f0_p2.shape:', f0_p2.shape)
        f1_p2 = self.reduction(z1_p2).squeeze(axis=3).squeeze(axis=2)
        # print('f1_p2.shape:', f1_p2.shape)
        f0_p3 = self.reduction(z0_p3).squeeze(axis=3).squeeze(axis=2)
        # print('f0_p3.shape:', f0_p3.shape)
        f1_p3 = self.reduction(z1_p3).squeeze(axis=3).squeeze(axis=2)
        # print('f1_p3.shape:', f1_p3.shape)
        f2_p3 = self.reduction(z2_p3).squeeze(axis=3).squeeze(axis=2)
        # print('f2_p3.shape:', f2_p3.shape)

        l_p1 = self.fc_id_2048_0(fg_p1)
        # print('l_p1.shape:', l_p1.shape)
        l_p2 = self.fc_id_2048_1(fg_p2)
        # print('l_p2.shape:', l_p2.shape)
        l_p3 = self.fc_id_2048_2(fg_p3)
        # print('l_p3.shape:', l_p3.shape)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        # print('l0_p2.shape:', l0_p2.shape)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        # print('l1_p2.shape:', l1_p2.shape)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        # print('l0_p3.shape:', l0_p3.shape)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        # print('l1_p3.shape:', l1_p3.shape)
        l2_p3 = self.fc_id_256_2_2(f2_p3)
        # print('l2_p3.shape:', l2_p3.shape)

        predict = paddle.concat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], axis=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3


if __name__ == '__main__':
    mgn = MGN()
    inputs = paddle.normal(shape=[1, 3, 384, 128])
    inputs.stop_gradient = False
    output = mgn(inputs)
    # print('output:', output)
