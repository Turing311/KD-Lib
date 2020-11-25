import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class MfnModel(nn.Module):

    
    def __init__(self):
        super(MfnModel, self).__init__()
#        global _weights_dict, _weights_dict_fc
#        _weights_dict = load_weights(weight_file)
#        _weights_dict_fc = load_weights(fc_file)

        self.conv1 = self.__conv(2, name='conv1', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.conv1_bn = self.__batch_normalization(2, 'conv1/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv1_dw = self.__conv(2, name='conv1_dw', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=64, bias=False)
        self.conv1_dw_bn = self.__batch_normalization(2, 'conv1_dw/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_ex = self.__conv(2, name='conv2_ex', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_ex_bn = self.__batch_normalization(2, 'conv2_ex/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_dw = self.__conv(2, name='conv2_dw', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), groups=128, bias=False)
        self.conv2_dw_bn = self.__batch_normalization(2, 'conv2_dw/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_em = self.__conv(2, name='conv2_em', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_em_bn = self.__batch_normalization(2, 'conv2_em/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_1_ex = self.__conv(2, name='conv2_1_ex', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_1_ex_bn = self.__batch_normalization(2, 'conv2_1_ex/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_1_dw = self.__conv(2, name='conv2_1_dw', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=128, bias=False)
        self.conv2_1_dw_bn = self.__batch_normalization(2, 'conv2_1_dw/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_1_em = self.__conv(2, name='conv2_1_em', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_1_em_bn = self.__batch_normalization(2, 'conv2_1_em/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_2_ex = self.__conv(2, name='conv2_2_ex', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_2_ex_bn = self.__batch_normalization(2, 'conv2_2_ex/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_2_dw = self.__conv(2, name='conv2_2_dw', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=128, bias=False)
        self.conv2_2_dw_bn = self.__batch_normalization(2, 'conv2_2_dw/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_2_em = self.__conv(2, name='conv2_2_em', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_2_em_bn = self.__batch_normalization(2, 'conv2_2_em/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_3_ex = self.__conv(2, name='conv2_3_ex', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_3_ex_bn = self.__batch_normalization(2, 'conv2_3_ex/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_3_dw = self.__conv(2, name='conv2_3_dw', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=128, bias=False)
        self.conv2_3_dw_bn = self.__batch_normalization(2, 'conv2_3_dw/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_3_em = self.__conv(2, name='conv2_3_em', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_3_em_bn = self.__batch_normalization(2, 'conv2_3_em/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_4_ex = self.__conv(2, name='conv2_4_ex', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_4_ex_bn = self.__batch_normalization(2, 'conv2_4_ex/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_4_dw = self.__conv(2, name='conv2_4_dw', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=128, bias=False)
        self.conv2_4_dw_bn = self.__batch_normalization(2, 'conv2_4_dw/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_4_em = self.__conv(2, name='conv2_4_em', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv2_4_em_bn = self.__batch_normalization(2, 'conv2_4_em/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_ex = self.__conv(2, name='conv3_ex', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_ex_bn = self.__batch_normalization(2, 'conv3_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_dw = self.__conv(2, name='conv3_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=256, bias=False)
        self.conv3_dw_bn = self.__batch_normalization(2, 'conv3_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_em = self.__conv(2, name='conv3_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_em_bn = self.__batch_normalization(2, 'conv3_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_1_ex = self.__conv(2, name='conv3_1_ex', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_1_ex_bn = self.__batch_normalization(2, 'conv3_1_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_1_dw = self.__conv(2, name='conv3_1_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv3_1_dw_bn = self.__batch_normalization(2, 'conv3_1_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_1_em = self.__conv(2, name='conv3_1_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_1_em_bn = self.__batch_normalization(2, 'conv3_1_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_2_ex = self.__conv(2, name='conv3_2_ex', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_2_ex_bn = self.__batch_normalization(2, 'conv3_2_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_2_dw = self.__conv(2, name='conv3_2_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv3_2_dw_bn = self.__batch_normalization(2, 'conv3_2_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_2_em = self.__conv(2, name='conv3_2_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_2_em_bn = self.__batch_normalization(2, 'conv3_2_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_3_ex = self.__conv(2, name='conv3_3_ex', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_3_ex_bn = self.__batch_normalization(2, 'conv3_3_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_3_dw = self.__conv(2, name='conv3_3_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv3_3_dw_bn = self.__batch_normalization(2, 'conv3_3_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_3_em = self.__conv(2, name='conv3_3_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_3_em_bn = self.__batch_normalization(2, 'conv3_3_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_4_ex = self.__conv(2, name='conv3_4_ex', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_4_ex_bn = self.__batch_normalization(2, 'conv3_4_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_4_dw = self.__conv(2, name='conv3_4_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv3_4_dw_bn = self.__batch_normalization(2, 'conv3_4_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_4_em = self.__conv(2, name='conv3_4_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_4_em_bn = self.__batch_normalization(2, 'conv3_4_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_5_ex = self.__conv(2, name='conv3_5_ex', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_5_ex_bn = self.__batch_normalization(2, 'conv3_5_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_5_dw = self.__conv(2, name='conv3_5_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv3_5_dw_bn = self.__batch_normalization(2, 'conv3_5_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_5_em = self.__conv(2, name='conv3_5_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_5_em_bn = self.__batch_normalization(2, 'conv3_5_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_6_ex = self.__conv(2, name='conv3_6_ex', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_6_ex_bn = self.__batch_normalization(2, 'conv3_6_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_6_dw = self.__conv(2, name='conv3_6_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv3_6_dw_bn = self.__batch_normalization(2, 'conv3_6_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_6_em = self.__conv(2, name='conv3_6_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv3_6_em_bn = self.__batch_normalization(2, 'conv3_6_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_ex = self.__conv(2, name='conv4_ex', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_ex_bn = self.__batch_normalization(2, 'conv4_ex/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_dw = self.__conv(2, name='conv4_dw', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), groups=512, bias=False)
        self.conv4_dw_bn = self.__batch_normalization(2, 'conv4_dw/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_em = self.__conv(2, name='conv4_em', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_em_bn = self.__batch_normalization(2, 'conv4_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_1_ex = self.__conv(2, name='conv4_1_ex', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_1_ex_bn = self.__batch_normalization(2, 'conv4_1_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_1_dw = self.__conv(2, name='conv4_1_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv4_1_dw_bn = self.__batch_normalization(2, 'conv4_1_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_1_em = self.__conv(2, name='conv4_1_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_1_em_bn = self.__batch_normalization(2, 'conv4_1_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_2_ex = self.__conv(2, name='conv4_2_ex', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_2_ex_bn = self.__batch_normalization(2, 'conv4_2_ex/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_2_dw = self.__conv(2, name='conv4_2_dw', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=256, bias=False)
        self.conv4_2_dw_bn = self.__batch_normalization(2, 'conv4_2_dw/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_2_em = self.__conv(2, name='conv4_2_em', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv4_2_em_bn = self.__batch_normalization(2, 'conv4_2_em/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv5_ex = self.__conv(2, name='conv5_ex', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv5_ex_bn = self.__batch_normalization(2, 'conv5_ex/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv5_dw = self.__conv(2, name='conv5_dw', in_channels=512, out_channels=512, kernel_size=(8, 8), stride=(1, 1), groups=512, bias=False)
        self.conv5_dw_bn = self.__batch_normalization(2, 'conv5_dw/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.fc1_512_1 = self.__dense(name = 'fc1_512_1', in_features = 512, out_features = 512, bias = False)
        self.bn_fc1_512 = self.__batch_normalization(2, 'bn_fc1_512', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.fc3_256_1 = self.__dense_fc(name = 'fc3_256_1', in_features = 256, out_features = 2510, bias = False)

    def forward(self, x, out_feature=False):
        conv1_pad       = F.pad(x, (1, 1, 1, 1))
        conv1           = self.conv1(conv1_pad)
        conv1_bn        = self.conv1_bn(conv1)
        relu_1          = F.relu(conv1_bn)
        conv1_dw_pad    = F.pad(relu_1, (1, 1, 1, 1))
        conv1_dw        = self.conv1_dw(conv1_dw_pad)
        conv1_dw_bn     = self.conv1_dw_bn(conv1_dw)
        relu_1_dw       = F.relu(conv1_dw_bn)
        conv2_ex        = self.conv2_ex(relu_1_dw)
        conv2_ex_bn     = self.conv2_ex_bn(conv2_ex)
        relu_2_ex       = F.relu(conv2_ex_bn)
        conv2_dw_pad    = F.pad(relu_2_ex, (1, 1, 1, 1))
        conv2_dw        = self.conv2_dw(conv2_dw_pad)
        conv2_dw_bn     = self.conv2_dw_bn(conv2_dw)
        relu_2_dw       = F.relu(conv2_dw_bn)
        conv2_em        = self.conv2_em(relu_2_dw)
        conv2_em_bn     = self.conv2_em_bn(conv2_em)
        conv2_1_ex      = self.conv2_1_ex(conv2_em_bn)
        conv2_1_ex_bn   = self.conv2_1_ex_bn(conv2_1_ex)
        relu_2_1_ex     = F.relu(conv2_1_ex_bn)
        conv2_1_dw_pad  = F.pad(relu_2_1_ex, (1, 1, 1, 1))
        conv2_1_dw      = self.conv2_1_dw(conv2_1_dw_pad)
        conv2_1_dw_bn   = self.conv2_1_dw_bn(conv2_1_dw)
        relu_2_1_dw     = F.relu(conv2_1_dw_bn)
        conv2_1_em      = self.conv2_1_em(relu_2_1_dw)
        conv2_1_em_bn   = self.conv2_1_em_bn(conv2_1_em)
        res2_1          = conv2_em_bn + conv2_1_em_bn
        conv2_2_ex      = self.conv2_2_ex(res2_1)
        conv2_2_ex_bn   = self.conv2_2_ex_bn(conv2_2_ex)
        relu_2_2_ex     = F.relu(conv2_2_ex_bn)
        conv2_2_dw_pad  = F.pad(relu_2_2_ex, (1, 1, 1, 1))
        conv2_2_dw      = self.conv2_2_dw(conv2_2_dw_pad)
        conv2_2_dw_bn   = self.conv2_2_dw_bn(conv2_2_dw)
        relu_2_2_dw     = F.relu(conv2_2_dw_bn)
        conv2_2_em      = self.conv2_2_em(relu_2_2_dw)
        conv2_2_em_bn   = self.conv2_2_em_bn(conv2_2_em)
        res2_2          = res2_1 + conv2_2_em_bn
        conv2_3_ex      = self.conv2_3_ex(res2_2)
        conv2_3_ex_bn   = self.conv2_3_ex_bn(conv2_3_ex)
        relu_2_3_ex     = F.relu(conv2_3_ex_bn)
        conv2_3_dw_pad  = F.pad(relu_2_3_ex, (1, 1, 1, 1))
        conv2_3_dw      = self.conv2_3_dw(conv2_3_dw_pad)
        conv2_3_dw_bn   = self.conv2_3_dw_bn(conv2_3_dw)
        relu_2_3_dw     = F.relu(conv2_3_dw_bn)
        conv2_3_em      = self.conv2_3_em(relu_2_3_dw)
        conv2_3_em_bn   = self.conv2_3_em_bn(conv2_3_em)
        res2_3          = res2_2 + conv2_3_em_bn
        conv2_4_ex      = self.conv2_4_ex(res2_3)
        conv2_4_ex_bn   = self.conv2_4_ex_bn(conv2_4_ex)
        relu_2_4_ex     = F.relu(conv2_4_ex_bn)
        conv2_4_dw_pad  = F.pad(relu_2_4_ex, (1, 1, 1, 1))
        conv2_4_dw      = self.conv2_4_dw(conv2_4_dw_pad)
        conv2_4_dw_bn   = self.conv2_4_dw_bn(conv2_4_dw)
        relu_2_4_dw     = F.relu(conv2_4_dw_bn)
        conv2_4_em      = self.conv2_4_em(relu_2_4_dw)
        conv2_4_em_bn   = self.conv2_4_em_bn(conv2_4_em)
        res2_4          = res2_3 + conv2_4_em_bn
        conv3_ex        = self.conv3_ex(res2_4)
        conv3_ex_bn     = self.conv3_ex_bn(conv3_ex)
        relu_3_ex       = F.relu(conv3_ex_bn)
        conv3_dw_pad    = F.pad(relu_3_ex, (1, 1, 1, 1))
        conv3_dw        = self.conv3_dw(conv3_dw_pad)
        conv3_dw_bn     = self.conv3_dw_bn(conv3_dw)
        relu_3_dw       = F.relu(conv3_dw_bn)
        conv3_em        = self.conv3_em(relu_3_dw)
        conv3_em_bn     = self.conv3_em_bn(conv3_em)
        conv3_1_ex      = self.conv3_1_ex(conv3_em_bn)
        conv3_1_ex_bn   = self.conv3_1_ex_bn(conv3_1_ex)
        relu_3_1_ex     = F.relu(conv3_1_ex_bn)
        conv3_1_dw_pad  = F.pad(relu_3_1_ex, (1, 1, 1, 1))
        conv3_1_dw      = self.conv3_1_dw(conv3_1_dw_pad)
        conv3_1_dw_bn   = self.conv3_1_dw_bn(conv3_1_dw)
        relu_3_1_dw     = F.relu(conv3_1_dw_bn)
        conv3_1_em      = self.conv3_1_em(relu_3_1_dw)
        conv3_1_em_bn   = self.conv3_1_em_bn(conv3_1_em)
        res3_1          = conv3_em_bn + conv3_1_em_bn
        conv3_2_ex      = self.conv3_2_ex(res3_1)
        conv3_2_ex_bn   = self.conv3_2_ex_bn(conv3_2_ex)
        relu_3_2_ex     = F.relu(conv3_2_ex_bn)
        conv3_2_dw_pad  = F.pad(relu_3_2_ex, (1, 1, 1, 1))
        conv3_2_dw      = self.conv3_2_dw(conv3_2_dw_pad)
        conv3_2_dw_bn   = self.conv3_2_dw_bn(conv3_2_dw)
        relu_3_2_dw     = F.relu(conv3_2_dw_bn)
        conv3_2_em      = self.conv3_2_em(relu_3_2_dw)
        conv3_2_em_bn   = self.conv3_2_em_bn(conv3_2_em)
        res3_2          = res3_1 + conv3_2_em_bn
        conv3_3_ex      = self.conv3_3_ex(res3_2)
        conv3_3_ex_bn   = self.conv3_3_ex_bn(conv3_3_ex)
        relu_3_3_ex     = F.relu(conv3_3_ex_bn)
        conv3_3_dw_pad  = F.pad(relu_3_3_ex, (1, 1, 1, 1))
        conv3_3_dw      = self.conv3_3_dw(conv3_3_dw_pad)
        conv3_3_dw_bn   = self.conv3_3_dw_bn(conv3_3_dw)
        relu_3_3_dw     = F.relu(conv3_3_dw_bn)
        conv3_3_em      = self.conv3_3_em(relu_3_3_dw)
        conv3_3_em_bn   = self.conv3_3_em_bn(conv3_3_em)
        res3_3          = res3_2 + conv3_3_em_bn
        conv3_4_ex      = self.conv3_4_ex(res3_3)
        conv3_4_ex_bn   = self.conv3_4_ex_bn(conv3_4_ex)
        relu_3_4_ex     = F.relu(conv3_4_ex_bn)
        conv3_4_dw_pad  = F.pad(relu_3_4_ex, (1, 1, 1, 1))
        conv3_4_dw      = self.conv3_4_dw(conv3_4_dw_pad)
        conv3_4_dw_bn   = self.conv3_4_dw_bn(conv3_4_dw)
        relu_3_4_dw     = F.relu(conv3_4_dw_bn)
        conv3_4_em      = self.conv3_4_em(relu_3_4_dw)
        conv3_4_em_bn   = self.conv3_4_em_bn(conv3_4_em)
        res3_4          = res3_3 + conv3_4_em_bn
        conv3_5_ex      = self.conv3_5_ex(res3_4)
        conv3_5_ex_bn   = self.conv3_5_ex_bn(conv3_5_ex)
        relu_3_5_ex     = F.relu(conv3_5_ex_bn)
        conv3_5_dw_pad  = F.pad(relu_3_5_ex, (1, 1, 1, 1))
        conv3_5_dw      = self.conv3_5_dw(conv3_5_dw_pad)
        conv3_5_dw_bn   = self.conv3_5_dw_bn(conv3_5_dw)
        relu_3_5_dw     = F.relu(conv3_5_dw_bn)
        conv3_5_em      = self.conv3_5_em(relu_3_5_dw)
        conv3_5_em_bn   = self.conv3_5_em_bn(conv3_5_em)
        res3_5          = res3_4 + conv3_5_em_bn
        conv3_6_ex      = self.conv3_6_ex(res3_5)
        conv3_6_ex_bn   = self.conv3_6_ex_bn(conv3_6_ex)
        relu_3_6_ex     = F.relu(conv3_6_ex_bn)
        conv3_6_dw_pad  = F.pad(relu_3_6_ex, (1, 1, 1, 1))
        conv3_6_dw      = self.conv3_6_dw(conv3_6_dw_pad)
        conv3_6_dw_bn   = self.conv3_6_dw_bn(conv3_6_dw)
        relu_3_6_dw     = F.relu(conv3_6_dw_bn)
        conv3_6_em      = self.conv3_6_em(relu_3_6_dw)
        conv3_6_em_bn   = self.conv3_6_em_bn(conv3_6_em)
        res3_6          = res3_5 + conv3_6_em_bn
        conv4_ex        = self.conv4_ex(res3_6)
        conv4_ex_bn     = self.conv4_ex_bn(conv4_ex)
        relu_4_ex       = F.relu(conv4_ex_bn)
        conv4_dw_pad    = F.pad(relu_4_ex, (1, 1, 1, 1))
        conv4_dw        = self.conv4_dw(conv4_dw_pad)
        conv4_dw_bn     = self.conv4_dw_bn(conv4_dw)
        relu_4_dw       = F.relu(conv4_dw_bn)
        conv4_em        = self.conv4_em(relu_4_dw)
        conv4_em_bn     = self.conv4_em_bn(conv4_em)
        conv4_1_ex      = self.conv4_1_ex(conv4_em_bn)
        conv4_1_ex_bn   = self.conv4_1_ex_bn(conv4_1_ex)
        relu_4_1_ex     = F.relu(conv4_1_ex_bn)
        conv4_1_dw_pad  = F.pad(relu_4_1_ex, (1, 1, 1, 1))
        conv4_1_dw      = self.conv4_1_dw(conv4_1_dw_pad)
        conv4_1_dw_bn   = self.conv4_1_dw_bn(conv4_1_dw)
        relu_4_1_dw     = F.relu(conv4_1_dw_bn)
        conv4_1_em      = self.conv4_1_em(relu_4_1_dw)
        conv4_1_em_bn   = self.conv4_1_em_bn(conv4_1_em)
        res4_1          = conv4_em_bn + conv4_1_em_bn
        conv4_2_ex      = self.conv4_2_ex(res4_1)
        conv4_2_ex_bn   = self.conv4_2_ex_bn(conv4_2_ex)
        relu_4_2_ex     = F.relu(conv4_2_ex_bn)
        conv4_2_dw_pad  = F.pad(relu_4_2_ex, (1, 1, 1, 1))
        conv4_2_dw      = self.conv4_2_dw(conv4_2_dw_pad)
        conv4_2_dw_bn   = self.conv4_2_dw_bn(conv4_2_dw)
        relu_4_2_dw     = F.relu(conv4_2_dw_bn)
        conv4_2_em      = self.conv4_2_em(relu_4_2_dw)
        conv4_2_em_bn   = self.conv4_2_em_bn(conv4_2_em)
        res4_2          = res4_1 + conv4_2_em_bn
        conv5_ex        = self.conv5_ex(res4_2)
        conv5_ex_bn     = self.conv5_ex_bn(conv5_ex)
        relu_5_ex       = F.relu(conv5_ex_bn)
        conv5_dw        = self.conv5_dw(relu_5_ex)
        conv5_dw_bn     = self.conv5_dw_bn(conv5_dw)
        fc1_512_0       = conv5_dw_bn.view(conv5_dw_bn.size(0), -1)
        fc1_512_1       = self.fc1_512_1(fc1_512_0)
        fc1_512_1       = fc1_512_1.reshape(-1, 512, 1, 1)
        bn_fc1_512      = self.bn_fc1_512(fc1_512_1)
        bn_fc1_512      = bn_fc1_512.reshape(bn_fc1_512.size()[0], bn_fc1_512.size()[1])
        slice_fc1, slice_fc2       = bn_fc1_512[:, :256], bn_fc1_512[:, 256:]
        eltwise_fc1 = torch.max(slice_fc1, slice_fc2)
        out       = self.fc3_256_1(eltwise_fc1)

        if out_feature == False:
            return out[:796]
        else:
            feature = eltwise_fc1.view(eltwise_fc1.size(0), -1)
            return out,feature

    def freeze(self):
        for name, p in self.named_parameters():
            p.requires_grad = False

        for name, p in self.fc3_256_1.named_parameters():
            p.requires_grad = True
    
    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

#        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
#        if 'bias' in _weights_dict[name]:
#            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        layer.weight.data.fill_(1)
        layer.bias.data.fill_(0)

#        if 'scale' in _weights_dict[name]:
#            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
#        else:
#            layer.weight.data.fill_(1)

#        if 'bias' in _weights_dict[name]:
#            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
#        else:
#            layer.bias.data.fill_(0)

#        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
#        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
#        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
#        if 'bias' in _weights_dict[name]:
#            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense_fc(name, **kwargs):
        layer = nn.Linear(**kwargs)
#        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict_fc[name]['weights']))
#        if 'bias' in _weights_dict_fc[name]:
#            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict_fc[name]['bias']))
        return layer
