from __future__ import annotations

from typing import Any
import pathlib
from datetime import datetime
import os
import yaml
import csv


class Logger(object):

    def __init__(self,
                 project_root_dir,
                 args,
                 algo_key='algo',
                 env_key='env_id',
                 hyperparameters_keys: list | None = None,
                 save_suffix: str = '',
                 additional_repos: list | None = None,
                 tensorboard_summary=False):

        self.project_root_dir = pathlib.Path(project_root_dir)
        high_run_dir = datetime.strftime(datetime.now(),
                                         f'%Y%m%d-{args[env_key]}')
        hparam_str_dict = dict()
        if hyperparameters_keys is not None:
            for key in hyperparameters_keys:
                hparam_str_dict[key] = args[key]

        hparam_str = ','.join([
            '%s=%s' % (k, str(hparam_str_dict[k]))
            for k in sorted(hparam_str_dict.keys())
        ])
        low_run_dir = datetime.strftime(datetime.now(),
                                        f'%H%M%S-{hparam_str}-{save_suffix}')
        self.run_dir = self.project_root_dir / 'logs' / args[
            algo_key] / high_run_dir / low_run_dir
        self.event_dir = self.run_dir / 'events'
        self.git_dir = self.run_dir / 'git'
        self.log_dir = self.run_dir / 'logs'

        self.event_dir.mkdir(exist_ok=True, parents=True)
        self.git_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.git_repo_paths = [self.project_root_dir]
        if additional_repos is not None:
            self.git_repo_paths.extend(additional_repos)

        self.record_git_diff()
        self.save_args(args)

        if tensorboard_summary:
            from tensorboardX import SummaryWriter
            self.summary_writer = SummaryWriter(log_dir=(self.event_dir),
                                                filename_suffix=low_run_dir)

    def save_args(self, args):
        with open(os.path.join(self.run_dir, 'train_params.yaml'), 'w') as fp:
            yaml.dump(args, fp, default_flow_style=False)


class TensorboardOrWandBLogger(object):

    def __init__(self,
                 logger: str = "wandb",
                 logger_instance: Any = None,
                 log_interval: int = 1):
        if logger == "wandb":
            from wandb.wandb_run import Run
            assert isinstance(
                logger_instance, Run
            ), "logger_instance must be an instance of wandb.wandb_run.Run"
            self.logger = logger_instance
        elif logger == "tensorboard":
            from tensorboardX import SummaryWriter
            assert isinstance(
                logger_instance, SummaryWriter
            ), "logger_instance must be an instance of tensorboardX.SummaryWriter"
            self.logger = logger_instance
        else:
            raise ValueError(f"Invalid logger: {logger}")
        self.logger_type = logger
        self.log_interval = log_interval

    def log(
        self,
        log_dict: dict,
        step: int,
        group: str = None,
    ):
        if step % self.log_interval != 0:
            return
        for key, value in log_dict.items():
            key = f"{group}/{key}" if group is not None else key
            if self.logger_type == "wandb":
                self.logger.log({key: value}, step=step)
            elif self.logger_type == "tensorboard":
                self.logger.add_scalar(
                    key,
                    value,
                    step,
                )
            else:
                raise ValueError(f"Invalid logger: {self.logger_type}")
        if self.logger_type == "tensorboard":
            self.logger.flush()





class RetLogger(object):

    def __init__(self, log_dir, fname=None):
        if fname is None:
            self.path = os.path.join(log_dir, 'log.csv')
        else:
            self.path = os.path.join(log_dir, fname)
        with open(self.path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'avg_ret', 'std_ret'])

    def log(self, step, avg_ret, std_ret):
        with open(self.path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, avg_ret, std_ret])
