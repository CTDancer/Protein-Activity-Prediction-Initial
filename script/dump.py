import os
import sys
import pprint
import pickle
import random

from tqdm import tqdm

import numpy as np

import torch

from torchdrug import core, models, tasks, datasets, utils, data
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import layer, model, dataset


def dump(cfg, dataset, task):
    dataloader = data.DataLoader(dataset, cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device(cfg.gpus[0])
    task = task.cuda(device)
    task.eval()
    preds = []
    for batch in tqdm(dataloader):
        batch = utils.cuda(batch, device=device)
        graph = batch["graph"]
        if task.graph_construction_model:
            graph = task.graph_construction_model(graph)
        output = task.model(graph, graph.node_feature.float())
        preds.append(output["graph_feature"].detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return pred


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

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

    if cfg.task['class'] == 'MultipleBinaryClassification':
        cfg.task.task = [_ for _ in range(len(dataset.tasks))]
    elif hasattr(dataset, "tasks"):
        cfg.task.task = dataset.tasks
    else:
        cfg.task.task = "placeholder"
    task = core.Configurable.load_config_dict(cfg.task)

    if cfg.get("model_checkpoint") is not None:
        model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        pretrained_dict = torch.load(model_checkpoint, map_location=torch.device('cpu'))
    else:
        model_checkpoint = os.path.expanduser(cfg.checkpoint)
        pretrained_dict = torch.load(model_checkpoint, map_location=torch.device('cpu'))['model']
    model_dict = task.state_dict()
    pretrained_dict_ = {}
    for k, v in model_dict.items():
        if k.startswith('mlp'): continue
        if k in model_dict:
            pretrained_dict_[k] = v
    task.load_state_dict(pretrained_dict_, strict=False)
    print("Loaded weights: ", pretrained_dict_.keys())

    pred = dump(cfg, dataset, task)
    with open(os.path.expanduser(cfg.output_file), "wb") as f:
        pickle.dump(pred, f)
        pickle.dump(dataset.pdb_files, f)