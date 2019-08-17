import chainer
from chainer import reporter as reporter_module
from chainer.training import extensions
from chainer import Variable
import chainer.functions as F
import copy
import numpy as np
import random

class ClassifierEvaluator(extensions.Evaluator):
    def get_batch(self, batch, xp):
        batchsize = len(batch)
        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y, dtype=xp.int32))
        return x_real, y_real
    
    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            x, y = self.get_batch(batch, target.xp)
            observation = {}
            with reporter_module.report_scope(observation):
                with chainer.no_backprop_mode():
                    y_pred = target(x)
                    loss = F.softmax_cross_entropy(y_pred, y)
                    acc = F.accuracy(y_pred, y)
                chainer.reporter.report({'loss': loss}, target)
                chainer.reporter.report({'accuracy': acc}, target)
            summary.add(observation)

        return summary.compute_mean()
