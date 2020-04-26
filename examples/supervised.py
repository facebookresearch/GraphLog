"""
Example template for defining a system.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.nn import RGCNConv

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule

from graphlog import GraphLog


class SupervisedRGCN(LightningModule):
    """
    Sample model to show how to define a template.
    """

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
        # init superclass
        super().__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout the model.
        """
        self.rgcn_layers = []
        for l in range(self.hparams.num_layers):
            in_channels = self.hparams.relation_embedding_dim
            out_channels = self.hparams.relation_embedding_dim
            num_bases = self.hparams.relation_embedding_dim

            self.rgcn_layers.append(
                RGCNConv(
                    in_channels,
                    out_channels,
                    self.hparams.num_classes,
                    num_bases,
                    root_weight=self.hparams.root_weight,
                    bias=self.hparams.bias,
                )
            )

        self.rgcn_layers = nn.ModuleList(self.rgcn_layers)
        self.classfier = []
        inp_dim = (
            self.hparams.relation_embedding_dim * 2
            + self.hparams.relation_embedding_dim
        )
        outp_dim = self.hparams.hidden_dim
        for l in range(self.hparams.classify_layers - 1):
            self.classfier.append(nn.Linear(inp_dim, outp_dim))
            self.classfier.append(nn.ReLU())
            inp_dim = outp_dim
        self.classfier.append(nn.Linear(inp_dim, self.hparams.num_classes))
        self.classfier = nn.Sequential(*self.classfier)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, batch):
        """
        We use random node embeddings for each forward call.
        """
        data = batch.graphs
        # initialize nodes randomly
        node_emb = torch.Tensor(
            size=(self.hparams.num_nodes, self.hparams.relation_embedding_dim)
        ).to(data.x.device)
        torch.nn.init.xavier_uniform_(node_emb, gain=1.414)
        x = F.embedding(data.x, node_emb)
        x = x.squeeze(1)

        # get edge attributes
        edge_types = data.edge_attr - 1
        for nr in range(self.hparams.num_layers - 1):
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)
            x = self.rgcn_layers[nr](x, data.edge_index, edge_types)
            x = F.relu(x)
        x = self.rgcn_layers[self.hparams.num_layers - 1](
            x, data.edge_index, edge_types
        )
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.num_nodes, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        # x = torch.cat(chunks, dim=0)
        # classify
        query_emb = []
        for i in range(len(chunks)):
            query = (
                batch.queries[i]
                .unsqueeze(0)
                .unsqueeze(2)
                .repeat(1, 1, chunks[i].size(2))
            )  # B x num_q x dim
            query_emb.append(torch.gather(chunks[i], 1, query))
        query_emb = torch.cat(query_emb, dim=0)
        query = query_emb.view(query_emb.size(0), -1)  # B x (num_q x dim)
        # pool the nodes
        # mean pooling
        node_avg = torch.cat(
            [torch.mean(chunks[i], 1) for i in range(len(chunks))], dim=0
        )  # B x dim
        # concat the query
        edges = torch.cat((node_avg, query), -1)  # B x (dim + dim x num_q)
        return self.classfier(edges)

    def loss(self, labels, logits):
        ce = F.cross_entropy(logits, labels)
        return ce

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        graphs = batch.graphs
        targets = batch.targets
        y_hat = self(batch)

        # calculate loss
        loss_val = self.loss(targets, y_hat)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        graphs = batch.graphs
        targets = batch.targets
        y_hat = self(batch)

        loss_val = self.loss(targets, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(targets == labels_hat).item() / (len(targets) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc,})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        log.info("Training data loader called.")
        gl = GraphLog()
        rule_world = gl.get_dataset_by_name(self.hparams.train_world)
        # when using multi-node (ddp) we need to add the  datasampler
        batch_size = self.hparams.batch_size

        loader = gl.get_dataloader_by_mode(
            rule_world, mode="train", batch_size=batch_size
        )
        return loader

    def val_dataloader(self):
        log.info("Validation data loader called.")
        gl = GraphLog()
        rule_world = gl.get_dataset_by_name(self.hparams.train_world)
        # when using multi-node (ddp) we need to add the  datasampler
        batch_size = self.hparams.batch_size

        loader = gl.get_dataloader_by_mode(
            rule_world, mode="valid", batch_size=batch_size
        )
        return loader

    def test_dataloader(self):
        log.info("Test data loader called.")
        gl = GraphLog()
        rule_world = gl.get_dataset_by_name(self.hparams.train_world)
        # when using multi-node (ddp) we need to add the  datasampler
        batch_size = self.hparams.batch_size

        loader = gl.get_dataloader_by_mode(
            rule_world, mode="test", batch_size=batch_size
        )
        return loader

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this during testing, similar to `validation_step`,
        with the data from the test dataloader passed in as `batch`.
        """
        output = self.validation_step(batch, batch_idx)
        # Rename output keys
        output["test_loss"] = output.pop("val_loss")
        output["test_acc"] = output.pop("val_acc")

        return output

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """
        results = self.validation_step_end(outputs)

        # rename some keys
        results["progress_bar"].update(
            {
                "test_loss": results["progress_bar"].pop("val_loss"),
                "test_acc": results["progress_bar"].pop("val_acc"),
            }
        )
        results["log"] = results["progress_bar"]
        results["test_loss"] = results.pop("val_loss")

        return results

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through `self.hparams`.
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument("--train_world", default="rule_0", type=str)
        parser.add_argument("--num_layers", default=2, type=int)
        parser.add_argument(
            "--num_classes", default=21, type=int, help="20 classes including UNK rel"
        )
        parser.add_argument("--relation_embedding_dim", default=100, type=int)
        parser.add_argument("--root_weight", default=False, action="store_true")
        parser.add_argument("--bias", default=False, action="store_true")
        parser.add_argument("--dropout", default=0.2, type=float)
        parser.add_argument("--classify_layers", default=2, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument("--hidden_dim", default=50, type=int)
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument(
            "--num_nodes", default=10000, type=int, help="Set a max number of nodes"
        )

        # data
        parser.add_argument(
            "--data_root", default=os.path.join(root_dir, "mnist"), type=str
        )

        # training params (opt)
        parser.add_argument("--epochs", default=20, type=int)
        parser.add_argument("--optimizer_name", default="adam", type=str)
        parser.add_argument("--batch_size", default=64, type=int)
        return parser
