import os, sys, time
import shutil
import yaml

import argparse
import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
import chainer.cuda
import chainer.links as L
from chainer import serializers

sys.path.append(os.path.dirname(__file__))
import util.yaml_utils as yaml_utils
import trainer.supervised as train

def load_pretrained_models(config, model_path):
    model = yaml_utils.load_model(config['func'], config['name'], config['args'])
    serializers.load_npz(model_path, model)
    return model

def calc_top5_acc(pred, t):
    top5_preds = pred.argsort()[:,-5:]
    return np.asarray(np.any(top5_preds.T == t, axis=0).mean(dtype='f'))

def calc_pred_entropy(pred):
    likelihood = F.softmax(pred, axis=1).data
    log_likelihood = F.log(likelihood)
    entropy = -F.sum(likelihood * log_likelihood, axis=1)
    return np.mean(entropy.data)

def main():
    schema = 'filename\tTop1\tTop5\tF-Score\tEntropy\tPrecision\tRecall\tF-score'
    parser = argparse.ArgumentParser(description='Target Model Tester \n ({})'.format(schema))
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--results_dir', type=str, default='./result/', help='directory to save the results to')
    parser.add_argument('--batchsize', type=int, default=128, help='Batchsize for testing')
    parser.add_argument('--process_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)


    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))
    pattern = "-".join([config.pattern, config.models['classifier']['name'], config.dataset['dataset_name']])
    out_path = args.results_dir + '/' + pattern

    # Model
    model_path = out_path + '/classifier{}.npz'.format(args.process_num)
    model = load_pretrained_models(config.models['classifier'], model_path)

    # Dataset
    test_dataset = yaml_utils.load_dataset(config, test=True)
    test_itr = chainer.iterators.SerialIterator(test_dataset, args.batchsize, repeat=False)

    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()  # Copy the model to the GPU

    xp = model.xp

    pred_labels = []
    correct_labels = []
    count = 0
    with chainer.using_config('train', False):
        for batch in test_itr:
            batchsize = len(batch)
            images = [batch[i][0] for i in range(batchsize)]
            labels = [batch[i][1] for i in range(batchsize)]
            x = xp.array(images)
            result = model(x).data
            pred_labels.append(chainer.cuda.to_cpu(result))
            correct_labels.append(np.array(labels))
            count += 1
            
    pred_labels = np.concatenate(pred_labels)
    correct_labels = np.concatenate(correct_labels)
    top1 = F.mean(F.accuracy(pred_labels,correct_labels)).data
    top5 = calc_top5_acc(pred_labels, correct_labels)
    precision, recall, Fscore, _ = F.classification_summary(pred_labels, correct_labels)
    out_results = {
        'test_{}'.format(args.process_num):{
            'accuracy': float(top1),
            'top-5 accuracy': float(top5),
            'precision' : float(F.mean(precision).data),
            'recall' : float(F.mean(recall).data),
            'f-score' : float(F.mean(Fscore).data)
        }
    }

    result_path = out_path + '/test_result.yaml'
    if os.path.exists(result_path):
        result_yaml = yaml.load(open(result_path, 'r+'), Loader=yaml.SafeLoader)
    else:
        result_yaml = {}
    result_yaml.update(out_results)
    with open(result_path, mode='w') as f:
        f.write(yaml.dump(result_yaml, default_flow_style=False))

    print('{}\t{}\t{}\t{}\t{}\t{}'.format(pattern, top1, top5, F.mean(precision).data, F.mean(recall).data, F.mean(Fscore).data))
    return out_results

if __name__ == '__main__':
    main()
