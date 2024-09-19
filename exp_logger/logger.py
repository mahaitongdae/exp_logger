from __future__ import annotations
import pathlib
from datetime import datetime
import git
import os
import yaml

class Logger(object):
    def __init__(self,
                 project_root_dir,
                 args,
                 algo_key='algo',
                 env_key='env_id',
                 hyperparameters_keys: list | None = None,
                 save_suffix: str = '',
                 additional_repos : list | None = None,
                 tensorboard_summary = False):

        self.project_root_dir = pathlib.Path(project_root_dir)
        high_run_dir = datetime.strftime(datetime.now(), f'%Y%m%d-{args[env_key]}')
        hparam_str_dict = dict()
        if hyperparameters_keys is not None:
            for key in hyperparameters_keys:
                hparam_str_dict[key] = args[key]

        hparam_str = ','.join(['%s=%s' % (k, str(hparam_str_dict[k])) for k in
                               sorted(hparam_str_dict.keys())])
        low_run_dir = datetime.strftime(datetime.now(), f'%H%M%S-{hparam_str}-{save_suffix}')
        self.run_dir = self.project_root_dir / 'logs' / args[algo_key] / high_run_dir / low_run_dir
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




    def record_git_diff(self):
        file_paths = []
        for repository_file_path in self.git_repo_paths:
            try:
                repo = git.Repo(repository_file_path, search_parent_directories=True)
            except Exception:
                print(f"Could not find git repository in {repository_file_path}. Skipping.")
                # skip if not a git repository
                continue
            # get the name of the repository
            repo_name = pathlib.Path(repo.working_dir).name
            t = repo.head.commit.tree
            diff_file_name = os.path.join(self.git_dir, f"{repo_name}.diff")
            # check if the diff file already exists
            if os.path.isfile(diff_file_name):
                continue
            # write the diff file
            print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
            with open(diff_file_name, "x") as f:
                content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
                f.write(content)
            # add the file path to the list of files to be uploaded
            file_paths.append(diff_file_name)
        return file_paths

    def save_args(self, args):
        with open(os.path.join(self.run_dir, 'train_params.yaml'), 'w') as fp:
            yaml.dump(args, fp, default_flow_style=False)