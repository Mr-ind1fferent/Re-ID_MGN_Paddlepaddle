import paddle
device = paddle.device.get_device()
paddle.device.set_device(device)
def extract_feature(model, loader):
    features = paddle.zeros([1], dtype='float32')
    flag = True

    for (inputs, labels) in loader:
        if device=='cuda:0':
            input_img = inputs.cuda()
        else:
            input_img = inputs.cpu()
        outputs = model(input_img)
        f1 = outputs[0].cpu()

        # flip
        # inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        # print(inputs.type)
        inputs = paddle.index_select(inputs,paddle.arange(inputs.shape[3] - 1, -1, -1),3 )
        if device=='cuda:0':
            input_img = inputs.cuda()
        else:
            input_img = inputs.cpu()

        outputs = model(input_img)
        # print('outputs:', outputs)
        f2 = outputs[0].cpu()
        # print('f1:', f1)
        # print('f2:', f2)
        ff = f1 + f2
        # print('ff:', ff)


        # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        fnorm = paddle.norm(ff, p=2, axis=1, keepdim=True)
        ff = paddle.divide(x=ff, y=fnorm.expand_as(ff))


        # print('features:', features)
        if flag:
            features = features + ff
            flag = False
        else:
            features = paddle.concat([features, ff], axis=0)
        # print('features:', features.shap
        features.stop_gradient = True

    return features
