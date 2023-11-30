import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics, utils
from torchdrug.core import Registry as R
from torchdrug.layers import functional


@R.register("tasks.PairReward")
class PairReward(tasks.Task, core.Configurable):

    _option_members = {"task"}

    def __init__(self, model, task=(), num_mlp_layer=1, graph_construction_model=None, verbose=0):
        super(PairReward, self).__init__()
        self.model = model
        self.task = task
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [1])

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        loss = F.binary_cross_entropy_with_logits(pred[..., 0] - pred[..., 1], target, reduction="mean")

        name = tasks._get_criterion_name("bce")
        if self.verbose > 0:
            for t, l in zip(self.task, loss):
                metric["%s [%s]" % (name, t)] = l
        metric[name] = loss
        all_loss += loss

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph1"]
        graph2 = batch["graph2"]
        if self.graph_construction_model:
            graph1 = self.graph_construction_model(graph1)
            graph2 = self.graph_construction_model(graph2)
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
        output2 = self.model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = torch.stack([
            self.mlp(output1["graph_feature"]), 
            self.mlp(output2["graph_feature"])
        ], dim=-1)
        return pred

    def target(self, batch):
        target = torch.ones((batch["graph1"].batch_size, 1), device=self.device)
        return target

    def evaluate(self, pred, target):
        metric = {}
        score = ((pred[..., 0] - pred[..., 1]) > 0).float().mean(dim=0)
        for t, s in zip(self.task, score):
            metric["accuracy [%s]" % (t)] = s

        return metric