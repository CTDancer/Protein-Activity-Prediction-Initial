import os
import sys
import math
import pprint
import pickle
import random

from tqdm import tqdm

import numpy as np

import torch
from torch import nn

from torchdrug import core, tasks, datasets, utils, metrics, data
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import model, dataset
from gearnet.dataset import bio_load_pdb


@torch.no_grad()
def retrieve(keys, query):
    cos_sim = nn.CosineSimilarity(dim=1)
    sim = -cos_sim(query.unsqueeze(-1), keys.transpose(0, 1).unsqueeze(0))   # (1, num_train)
    retrieval_items = sim.argsort(dim=1)     # (num_test, k)
        
    return retrieval_items, sim


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

    logger = util.get_root_logger(file=False)
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)

    if cfg.task['class'] == 'MultipleBinaryClassification':
        cfg.task.task = [_ for _ in range(len(dataset.tasks))]
    elif hasattr(dataset, "tasks"):
        cfg.task.task = dataset.tasks
    else:
        cfg.task.task = "placeholder"
    task = core.Configurable.load_config_dict(cfg.task)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        pretrained_dict = {}
        for k, v in model_dict['model'].items():
            if k in task.state_dict():
                pretrained_dict[k] = v
        task.load_state_dict(pretrained_dict, strict=False)
        print("Loaded weights: ", pretrained_dict.keys())

    with open(os.path.expanduser(cfg.output_file), "rb") as f:
        reprs = pickle.load(f)
        pdb_files = pickle.load(f)
    keys = torch.as_tensor(reprs)
    keys = keys.cuda()

    device = torch.device(cfg.gpus[0])
    task = task.cuda(device)
    task.eval()

    # pdb_path = "~/scratch/protein-datasets/ABE8e.pdb"
    pdb_path = "~/scratch/protein-datasets/TadA005v1723.pdb"
    pdb_path = os.path.expanduser(pdb_path)
    protein, sequence = bio_load_pdb(pdb_path)
    protein = data.Protein.pack([protein])
    protein = protein.cuda(device)
    protein.view = "residue"
    if task.graph_construction_model:
        graph = task.graph_construction_model(protein)
    query = task.model(graph, graph.node_feature.float())["graph_feature"]
    retrieval_items, sim = retrieve(keys, query)
    retrieval_items = retrieval_items.squeeze(0)
    
    for item in retrieval_items.unbind():
        print(pdb_files[item], sim[0, item])