import csv
import logging
import os
import sys
sys.path.append('..')
from collections import OrderedDict
from pathlib import Path

import numpy as np
import wandb
import yaml

class ExpHandler:
    _home = Path.home()

    def __init__(self, en_wandb=False, args=None):
        project_name = os.getenv('WANDB_PROJECT', default='default_project')
        exp_name = os.getenv('exp_name', default='default_group')
        run_name = os.getenv('run_name', default='default_name')
        self._exp_id = f'{self._get_exp_id()}_{run_name}'
        self._exp_name = exp_name

        if args.resume != '' and (Path(args.resume) / 'config.yaml').exists():
            print('----------resuming-----------')
            self._save_dir = Path(args.resume)
            with open(self._save_dir / 'config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            if args.strict_resume:
                self.resume_sanity(args, config)
            if args.en_wandb:
                self.wandb_run = wandb.init(project=project_name, group=exp_name, name=run_name, save_code=True,
                                            id=config['wandb_id'], resume='allow')
        else:
            self._save_dir = os.path.join('{}/.exp/{}'.format(self._home, os.getenv('WANDB_PROJECT', default='default_project')),
                                      exp_name, self._exp_id)
            if not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir)
            if en_wandb:
                self.wandb_run = wandb.init(project=project_name, group=exp_name, name=run_name,settings=wandb.Settings(start_method="fork"))

        sym_dest = self._get_sym_path('N')
        os.symlink(self._save_dir, sym_dest)

        self._logger = self._init_logger()
        self._en_wandb = en_wandb


    @staticmethod
    def resume_sanity(args, old_conf):
        print('-' * 10, 'Resume sanity check', '-' * 10)
        old_config_hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in old_conf.items()
                                if k not in args.resume_check_exclude_keys}
        new_config_hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in vars(args).items()
                                if k not in args.resume_check_exclude_keys}
        print(f'Diff config: {set(old_config_hashable.items()) ^ set(new_config_hashable.items())}')
        assert old_config_hashable == new_config_hashable, 'Resume sanity check failed'

    def _get_sym_path(self, state):
        sym_dir = f'{self._home}/.exp/syms'
        if not os.path.exists(sym_dir):
            os.makedirs(sym_dir)

        sym_dest = os.path.join(sym_dir, '--'.join([self._exp_id, state, self._exp_name]))
        return sym_dest

    @property
    def save_dir(self):
        return self._save_dir

    @staticmethod
    def _get_exp_id():
        with open(f'{ExpHandler._home}/.core/counter', 'r+') as f:
            counter = eval(f.read())
            f.seek(0)
            f.write(str(counter + 1))
        with open(f'{ExpHandler._home}/.core/identifier', 'r+') as f:
            identifier = f.read()[0]
        exp_id = '{}{:04d}'.format(identifier, counter)
        return exp_id

    def _init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(self._save_dir, f'{self._exp_id}_log.txt'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def save_config(self, args):
        conf = vars(args)
        conf['exp_id'] = self._exp_id
        conf['commit'] = os.getenv('commit', default='not_set')
        conf['run_id'] = self._exp_id.split('_')[0]
        if hasattr(self, 'wandb_run'):
            conf['wandb_id'] = self.wandb_run.id
        with open(f'{self._save_dir}/config.yaml', 'w') as f:
            yaml.dump(conf, f)

        if self._en_wandb:
            wandb.config.update(conf,allow_val_change=True)

    def write(self, prefix, eval_metrics=None, train_metrics=None, **kwargs):
        rowd = OrderedDict([(f'{prefix}/{k}', v) for k, v in kwargs.items() ])
        if eval_metrics:
            rowd.update([(f'{prefix}/eval_' + k, v) for k, v in eval_metrics.items()])
        if train_metrics:
            rowd.update([(f'{prefix}/train_' + k, v) for k, v in train_metrics.items()])

        path = os.path.join(self._save_dir, f'{self._exp_id}_{prefix}_summary.csv')
        initial = not os.path.exists(path)
        with open(path, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=rowd.keys())
            if initial:
                dw.writeheader()
            dw.writerow(rowd)

        if self._en_wandb:
            wandb.log(rowd)

    def log(self, msg):
        self._logger.info(msg)

    def finish(self):
        Path(f'{self._save_dir}/finished').touch()
        os.rename(self._get_sym_path('N'), self._get_sym_path('Y'))

def consume_prefix_in_state_dict_if_present(
        state_dict, prefix
):
    r"""Strip the prefix in state_dict in place, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count
