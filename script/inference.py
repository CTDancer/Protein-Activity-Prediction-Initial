import os
import sys
import glob
import math
import pprint
import random

from tqdm import tqdm

import numpy as np

import torch
from torch.optim import lr_scheduler

from torchdrug import core, models, tasks, datasets, utils, data
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import dataset, model
from gearnet.dataset import bio_load_pdb


def test(cfg, solver):
    solver.evaluate("valid")
    return solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    task = core.Configurable.load_config_dict(cfg.task)
    task.preprocess(dataset, None, None)
    transform = core.Configurable.load_config_dict(cfg.transform)

    if cfg.get("checkpoint") is not None:
        cfg.checkpoint = os.path.expanduser(cfg.checkpoint)
        pretrained_dict = torch.load(cfg.checkpoint, map_location=torch.device('cpu'))['model']
        task.load_state_dict(pretrained_dict)

    dataset_path = os.path.expanduser(cfg.dataset_path)
    pdb_files = sorted(glob.glob(os.path.join(dataset_path, "*.pdb")))

    device = torch.device(cfg.gpu)
    task = task.cuda(device)
    task.eval()
    batch_size = cfg.get("batch_size", 1)
    preds = []
    for i in tqdm(range(0, len(pdb_files), batch_size)):
        proteins = []
        for pdb_file in pdb_files[i:i+batch_size]:
            protein, sequence = bio_load_pdb(pdb_file)
            proteins.append(protein)
        protein = data.Protein.pack(proteins)
        protein = protein.cuda(device)
        batch = {"graph": protein}
        batch = transform(batch)
        with torch.no_grad():
            pred = task.predict(batch)
        for j, value in enumerate(pred.cpu().unbind()):
            name = os.path.basename(pdb_files[i+j])[:-4]
            preds.append((name, value.item()))

    preds = sorted(preds, key=lambda x: -x[-1])
    print(preds)
