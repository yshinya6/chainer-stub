import os, sys, time
import argparse
import yaml
import shutil
import subprocess
import numpy as np
import logging

sys.path.append(os.path.dirname(__file__))
import util.yaml_utils as yaml_utils
from util.slack import SlackWrapper

SEED = [42, 7, 19, 61, 3, 17, 23, 37, 11, 71]

logging.basicConfig(filename='expr.log',format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger("Experiment-Manager")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(name)s|%(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)


def make_result_dir_path(config, result_dir):
    model_name = config.models['classifier']['name']
    pattern = "-".join([config.pattern, model_name, config.dataset['dataset_name']])
    result_path = result_dir + pattern
    return result_path, pattern


def make_summary(result_dir, pattern):
    result_path = result_dir + '/test_result.yaml'
    result = yaml.load(open(result_path), Loader=yaml.SafeLoader)
    accs = []

    for key in result:
        accs.append(result[key]['accuracy'])

    summary = {
        'summary': {
            'pattern': pattern,
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs))
        }
    }
    result.update(summary)

    with open(result_path, mode='w') as f:
        f.write(yaml.dump(result, default_flow_style=False))

    return summary


def train(config_path, result_dir):
    config = yaml_utils.Config(yaml.load(open(config_path), Loader=yaml.SafeLoader))
    result_dir, pattern = make_result_dir_path(config, result_dir)

    # Setup the training
    iteration = config.expr_itr
    main_script = config.main
    train_cmd = ['mpiexec', '--allow-run-as-root', '-n', '8', 'python', main_script, '--config_path', config_path]

    logger.info("Start Training: {} for {} iterations".format(pattern, iteration))
    for i, seed in enumerate(SEED[:iteration]):
        cmd =  train_cmd + ['--seed', str(seed), '--process_num', str(i)]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            logger.info("Succ Train: Iteration no.{} of {}".format(i+1, pattern))
        else:
            logger.info("Fail Train: Iteration no.{} of {}".format(i+1, pattern))
            logger.info("ErrorLog: {}".format(ret.stderr))
            break

    logger.info("End Training: {} for {} iterations".format(pattern, iteration))


def test(config_path, result_dir):
    config = yaml_utils.Config(yaml.load(open(config_path), Loader=yaml.SafeLoader))
    result_dir, pattern = make_result_dir_path(config, result_dir)

    # Setup
    iteration = config.expr_itr
    main_script = config.main.replace('trainer','tester')
    test_cmd = ['python', main_script, '--config_path', config_path]

    logger.info("Start Testing: {} for {} iterations".format(pattern, iteration))
    for i in range(iteration):
        cmd = test_cmd + ['--process_num', str(i)]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            logger.info("Succ Test: Iteration no.{} of {}".format(i+1, pattern))
        else:
            logger.info("Fail Test: Iteration no.{} of {}".format(i+1, pattern))
            logger.info("ErrorLog: {}".format(ret.stderr))

    make_summary(result_dir, pattern)

    logger.info("End Testing: {} for {} iterations".format(pattern, iteration))


def expr(config_path, result_dir):
    reporter = SlackWrapper()

    config = yaml_utils.Config(yaml.load(open(config_path), Loader=yaml.SafeLoader))
    result_dir, pattern = make_result_dir_path(config, result_dir)

    # Setup the training
    iteration = config.expr_itr
    main_script = config.main
    train_cmd = ['mpiexec', '--allow-run-as-root', '-n', '8', 'python', main_script, '--config_path', config_path]
    test_cmd = ['python', main_script.replace('trainer','tester'), '--config_path', config_path]

    fail_flag = False

    logger.info("Start: {} for {} iterations".format(pattern, iteration))
    for i, seed in enumerate(SEED[:iteration]):
        # Training
        cmd =  train_cmd + ['--seed', str(seed), '--process_num', str(i)]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            logger.info("Succ Train: Iteration no.{} of {}".format(i+1, pattern))
        else:
            fail_flag = True
            logger.info("Fail Train: Iteration no.{} of {}".format(i+1, pattern))
            logger.info("ErrorLog: {}".format(ret.stderr))
            break

        # Testing
        cmd = test_cmd + ['--process_num', str(i)]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            logger.info("Succ Test: Iteration no.{} of {}".format(i+1, pattern))
        else:
            logger.info("Fail Test: Iteration no.{} of {}".format(i+1, pattern))
            logger.info("ErrorLog: {}".format(ret.stderr))

    if fail_flag:
        msg = "{} was failed while the training.".format(pattern)
        logger.info(msg)
        reporter.report_fail(msg)
    else:
        summary = make_summary(result_dir, pattern)
        logger.info("End: {} for {} iterations".format(pattern, iteration))
        reporter.report_summary(summary)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()

    p_run = sub.add_parser('run')
    p_run.add_argument('--config_path', type=str, nargs='+', help='paths to config file')
    p_run.add_argument('--results_dir', type=str, default='./result/',
                        help='directory to save the results to')
    p_run.set_defaults(func=expr)

    p_train = sub.add_parser('train')
    p_train.add_argument('--config_path', type=str, nargs='+', help='paths to config file')
    p_train.add_argument('--results_dir', type=str, default='./result/',
                        help='directory to save the results to')
    p_train.set_defaults(func=train)
    
    p_test = sub.add_parser('test')
    p_test.add_argument('--config_path', type=str, nargs='+', help='paths to config file')
    p_test.add_argument('--results_dir', type=str, default='./result/',
                        help='directory to save the results to')
    p_test.set_defaults(func=test)

    args = parser.parse_args()

    for config_path in args.config_path:
        args.func(config_path, args.results_dir)


if __name__ == '__main__':
    main()
