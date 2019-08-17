

#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
import sys, os
sys.path.append(os.path.dirname(__file__))
from updater.updater import AbstractUpdater

class ClassifierUpdater(AbstractUpdater):
    def __init__(self, *args, **kwargs):
        self.classifier = kwargs.pop('classifier')
        self.loss = F.softmax_cross_entropy
        super(ClassifierUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        classifier = self.classifier
        optimizer = self.get_optimizer('main')
        xp = classifier.xp
        x, y_real = self.get_batch(xp)
        y_pred = classifier(x)
        loss = self.loss(y_pred, y_real)
        acc = F.accuracy(y_pred, y_real)
        chainer.reporter.report({'loss': loss}, self.classifier)
        chainer.reporter.report({'accuracy': acc}, self.classifier)
        classifier.cleargrads()
        loss.backward()
        optimizer.update()
