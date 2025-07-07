
import os
import shutil
import argparse
import numpy as np
import pandas as pd

from .execution import execute_script


def normalized_performance_score(s):
    return 1 / (1 + s)


def comprehensive_score(cr, nps):
    return 0.5 * cr + 0.5 * nps


def evaluate_code(path, task, metric, idx, device=0, n_trials=5):
    is_pass = False
    workspace = f"{path}/{task}"
    # Create evaluation path
    eval_dir = f"{workspace}/evaluation"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    eval_fname = f"{eval_dir}/{task}.csv"

    pass_trials = []
    filename = f"{task}_{idx}.py"
    print(f"...executing: {filename}", end="-->")
    rcode, log = execute_script(filename, work_dir=workspace, device=device)
    if rcode == 0 and "Model Performance on Test Set".lower() in log.lower():
        is_pass = True
        print("passed ^_^!", log.split("\n")[-3], log.split("\n")[-2])
    elif rcode == 0 and "Model Performance on Test Set".lower() not in log.lower():
        is_pass = False
        print("no error, but incomplete!")
    else:
        lastline = log.split("\n")[-2]
        if "ModuleNotFoundError" in lastline:
            print(workspace + "/" + filename)
            print(lastline)
        print('failed T_T!')
        is_pass = False

    return is_pass, workspace + "/" + filename
