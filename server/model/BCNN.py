import model.feature_extractor as fe
import torchvision
import torch
import torch.nn as nn
import functools
import operator
from model.compact_bilinear_pooling import CountSketch
from torch.autograd import Function
from model.matrixSquareRoot import MatrixSquareRoot 
import torch.nn.functional as F

matrix_sqrt = MatrixSquareRoot.apply

def create_backbone(model_name, finetune_model=True, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'vgg':
        """ VGG
        """
        model_ft = fe.VGG() 
        set_parameter_requires_grad(model_ft, finetune_model)

        output_dim = 512
    else:
        print("Invalid model name, exiting...")
        # logger.debug("Invalid mode name")
        exit()

    return model_ft, output_dim

def set_parameter_requires_grad(model, requires_grad):
    if requires_grad:
        for param in model.parameters():
            param.requires_grad = True


class BCNNModule(nn.Module):
    def __init__(self, num_classes, feature_extractors=None,
            pooling_fn=None, order=2, m_sqrt_iter=0, demo_agg=False,
            fc_bottleneck=False, learn_proj=False):
        super(BCNNModule, self).__init__()

        assert feature_extractors is not None
        assert pooling_fn is not None

        self.feature_extractors = feature_extractors
        self.pooling_fn = pooling_fn 

        self.feature_dim = self.pooling_fn.get_output_dim()
        if fc_bottleneck:
            self.fc = nn.Sequential(nn.Linear(self.feature_dim, 1024, bias=True), 
                                    nn.Linear(1024, num_classes, bias=True))
        else:
            self.fc = nn.Linear(self.feature_dim, num_classes, bias=True) 

        if m_sqrt_iter > 0:
            self.m_sqrt = MatrixSquareRoot(
                m_sqrt_iter,
                int(self.feature_dim ** 0.5),
                backwardIter=5
            )
        else:
            self.m_sqrt = None

        self.demo_agg = demo_agg
        self.order = order
        self.learn_proj = learn_proj

    def get_order(self):
        return self.order

    def forward(self, *args):
        x = self.feature_extractors(*args)

        bs, _, h1, w1 = x[0].shape
        for i in range(1, len(args)):
            _, _, h2, w2 = x[i].shape
            if h1 != h2 or w1 != w2:
                x[i] = torch.nn.functional.interpolate(x[i], size=(h1, w1),
                                                    mode='bilinear')
        z = self.pooling_fn(*x)

        if self.m_sqrt is not None:
            z = self.m_sqrt(z)
        z = z.view(bs, self.feature_dim)
        z = torch.sqrt(F.relu(z) + 1e-5) - torch.sqrt(F.relu(-z) + 1e-5)
        z = torch.nn.functional.normalize(z)

        # linear classifier
        y = self.fc(z)

        return y


class MultiStreamsCNNExtractors(nn.Module):
    def __init__(self, backbones_list, dim_list, proj_dim=0):
        super(MultiStreamsCNNExtractors, self).__init__()
        self.feature_extractors = nn.ModuleList(backbones_list)
        if proj_dim > 0:
            temp = [nn.Sequential(x, \
                        nn.Conv2d(fe_dim, proj_dim, 1, 1, bias=False)) \
                        for x, fe_dim in zip(self.feature_extractors, dim_list)]
            self.feature_extractors = nn.ModuleList(temp)

class BCNN_sharing(MultiStreamsCNNExtractors):
    def __init__(self, backbones_list, dim_list, proj_dim=0, order=2):
        super(BCNN_sharing, self).__init__(backbones_list, dim_list, proj_dim)

        # one backbone network for sharing parameters
        assert len(backbones_list) == 1 

        self.order = order

    def get_number_output(self):
        return self.order

    def forward(self, *args):
        # y = self.feature_extractors[0](x)
        y = [self.feature_extractors[0](x) for x in args]

        if len(args) == 1:
            # out = y * self.order
            # y[0].register_hook(lambda grad: print(grad[0,0,:3,:3]))

            # return out
            return y * self.order
            # return [y for z in range(self.order)] 
        else:
            return y


class TensorProduct(nn.Module):
    def __init__(self, dim_list):
        super(TensorProduct, self).__init__()
        self.output_dim = functools.reduce(operator.mul, dim_list)

        # Use tensor sketch for the order greater than 2
        assert len(dim_list) == 2

    def get_output_dim(self):
        return self.output_dim

    def forward(self, *args):
        (x1, x2) = args
        [bs, c1, h1, w1] = x1.size()
        [bs, c2, h2, w2] = x2.size()
        
        x1 = x1.view(bs, c1, h1*w1)
        x2 = x2.view(bs, c2, h2*w2)
        y = torch.bmm(x1, torch.transpose(x2, 1, 2))

        # return y.view(bs, c1*c2) / (h1 * w1)
        return y / (h1 * w1)


def create_bcnn_model(model_names_list, num_classes,
                pooling_method='outer_product', fine_tune=True, pre_train=True,
                embedding_dim=8192, order=2, m_sqrt_iter=0,
                fc_bottleneck=False, proj_dim=0, update_sketch=False,
                gamma=0.5):


    temp_list = [create_backbone(model_name, finetune_model=fine_tune, \
            use_pretrained=pre_train) for model_name in model_names_list]


    temp_list = list(map(list, zip(*temp_list)))
    backbones_list = temp_list[0]


    # list of feature dimensions of the backbone networks
    dim_list = temp_list[1]
    # BCNN mdoels with sharing parameters. The computation of the two backbone
    # networks are shared resulting in a symmetric BCNN
    
    dim_list = dim_list * order
    feature_extractors = BCNN_sharing(
            backbones_list,
            dim_list,
            proj_dim, order
    )

    # update the reduced feature dimension in dim_list if there is
    # dimensionality reduction
    if proj_dim > 0:
        dim_list = [proj_dim for x in dim_list]

    if pooling_method == 'outer_product':
        pooling_fn = TensorProduct(dim_list)
    else:
        raise ValueError('Unknown pooling method: %s' % pooling_method)

    learn_proj = True if proj_dim > 0 else False
    return BCNNModule(
            num_classes,
            feature_extractors,
            pooling_fn,
            order,
            m_sqrt_iter=m_sqrt_iter,
            fc_bottleneck=fc_bottleneck,
            learn_proj=learn_proj
    )


