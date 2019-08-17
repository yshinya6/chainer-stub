import os, sys, time
import shutil
import yaml

import argparse
import chainer
from chainer import links as L
from chainer import training
from chainer.training import extension
from chainer.training import extensions
import chainermn
from mpi4py import MPI
from chainer.training.triggers import MaxValueTrigger, ManualScheduleTrigger

sys.path.append(os.path.dirname(__file__))
import util.yaml_utils as yaml_utils
from evaluator.classification import ClassifierEvaluator


def create_result_dir(result_dir, config_path, config):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

    copy_to_result_dir(config_path, result_dir)
    

def load_models(model_config):
    model = yaml_utils.load_model(model_config['func'], model_config['name'], model_config['args'])
    if 'pretrained' in model_config:
        chainer.serializers.load_npz(model_config['pretrained'], model)
    return model


def make_optimizer(model, comm, config):
    # Select from https://docs.chainer.org/en/stable/reference/optimizers.html.
    # NOTE: The order of the arguments for optimizers follows their definitions.
    opt_algorithm = yaml_utils.load_optimizer(config.optimizer['algorithm'], args=config.optimizer['args'])
    optimizer = chainermn.create_multi_node_optimizer(opt_algorithm, comm)
    optimizer.setup(model)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--results_dir', type=str, default='./result/',
                        help='directory to save the results to')
    parser.add_argument('--resume', type=str, default='',
                        help='path to the snapshot')
    parser.add_argument('--process_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    
    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))
    pattern = "-".join([config.pattern, config.models['classifier']['name'], config.dataset['dataset_name']])
    comm = chainermn.create_communicator()
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(MPI.COMM_WORLD.Get_size()))
        print('Num Minibatch-size: {}'.format(config.batchsize))
        print('Num Epoch: {}'.format(config.epoch))
        print('==========================================')

    # Model
    classifier = load_models(config.models['classifier'])
    
    if args.resume:
        print("Resume training with snapshot:{}".format(args.resume))
        chainer.serializers.load_npz(args.resume, classifier)
    
    chainer.cuda.get_device_from_id(device).use()
    classifier.to_gpu()
    # models = {"classifier": classifier}
    
    # Optimizer
    opt = make_optimizer(classifier, comm, config)
    opt.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Dataset
    if comm.rank == 0:
        dataset = yaml_utils.load_dataset(config)
        first_size = int(len(dataset) * config.train_val_split_ratio)
        train, val = chainer.datasets.split_dataset_random(dataset, first_size, seed=args.seed)
    else:
        yaml_utils.load_module(config.dataset['dataset_func'], config.dataset['dataset_name'])
        train, val = None, None

    train = chainermn.scatter_dataset(train, comm)
    val = chainermn.scatter_dataset(val, comm)

    # Iterator
    train_iterator = chainer.iterators.SerialIterator(train, config.batchsize)
    val_iterator = chainer.iterators.SerialIterator(val, config.batchsize, repeat=False, shuffle=False)
    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'classifier': classifier,
        'iterator': train_iterator,
        'optimizer': opt,
        'device': device,
    })

    # Updater
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)
    out = args.results_dir + '/' + pattern

    if comm.rank == 0:
        create_result_dir(out, args.config_path, config)
    
    # Trainer
    trainer = training.Trainer(updater, (config.epoch, 'epoch'), out=out)

    # Evaluator
    evaluator = ClassifierEvaluator(val_iterator, classifier, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Learning Rate Schedule (fixed)
    schedule = [config.epoch*0.3, config.epoch*0.6, config.epoch*0.8]
    trainer.extend(extensions.ExponentialShift('lr', 0.1),   
               trigger=ManualScheduleTrigger(schedule,'epoch'))


    report_keys = ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']
    if comm.rank == 0:
        # Set up logging
        trainer.extend(extensions.snapshot_object(classifier, 'classifier{}.npz'.format(args.process_num)), trigger=MaxValueTrigger('validation/main/accuracy'))
        trainer.extend(extensions.LogReport(keys=report_keys,trigger=(config.display_interval, 'epoch')))
        trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'epoch'))
        trainer.extend(extensions.ProgressBar(update_interval=config.progressbar_interval))
    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
