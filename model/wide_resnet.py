
import chainer
import chainer.functions as F
import chainer.links as L

def conv3x3(in_channels, out_channels, stride=1):
    w = chainer.initializers.HeNormal()
    return L.Convolution2D(in_channels, out_channels, ksize=3, stride=stride, pad=1, initialW=w, nobias=True)


class WideBasicBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(WideBasicBlock, self).__init__()
        w = chainer.initializers.HeNormal()
        self.dropout_rate = dropout_rate

        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.Convolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1, initialW=w, nobias=True)
            self.bn2 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(out_channels, out_channels, ksize=3, stride=stride, pad=1, initialW=w, nobias=True)

            if stride != 1 or in_channels != out_channels:
                self.shortcut = L.Convolution2D(in_channels, out_channels, ksize=1, stride=stride, nobias=True)
            else:
                self.shortcut = F.identity

    def __call__(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.dropout(out, ratio=self.dropout_rate)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet(chainer.Chain):
    def __init__(self, depth, widen_factor, class_labels, dropout_rate):
        super(WideResNet, self).__init__()
        k = widen_factor
        self.in_channels = 16
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6

        nStages = [16, 16*k, 32*k, 64*k]

        with self.init_scope():
            self.conv1 = conv3x3(3,nStages[0])
            self.layer1 = self._make_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
            self.layer2 = self._make_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
            self.layer3 = self._make_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
            self.bn1 = L.BatchNormalization(nStages[3], decay=0.9)
            self.fc = L.Linear(nStages[3], class_labels)

    def _make_layer(self, block, out_channels, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
            self.in_channels = out_channels

        return chainer.Sequential(*layers)

    def __call__(self, x, feature_extract=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        if(feature_extract):
            return out
        out = F.average_pooling_2d(out, 8)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out


class WRN_40_10(WideResNet):
    def __init__(self, class_labels, dropout_rate=0.3):
        super().__init__(40, 10, class_labels=int(class_labels), dropout_rate=dropout_rate)