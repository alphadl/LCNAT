# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.trainer import Trainer
from fairseq.criterions import FairseqCriterion, register_criterion, nat_loss
from torch import Tensor

import numpy as np
from fairseq import utils
import h5py

from . import FairseqCriterion, register_criterion, LabelSmoothedDualImitationCriterion

@register_criterion("lcnat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smooting, lcnat_weight_path, lcnat_src_vocab_path, lcnat_tgt_vocab_path):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.lcnat_weight_path = lcnat_weight_path
        self.lcnat_src_vocab_path = lcnat_src_vocab_path
        self.lcnat_tgt_vocab_path = lcnat_tgt_vocab_path

        # load the pretrained weight
        assert self.lcnat_weight_path is not None
        assert self.lcnat_src_vocab_path is not None
        assert self.lcnat_tgt_vocab_path is not None

        fw = h5py.File(self.lcnat_weight_path, 'r')
        lw = fw['weights']
        self.lcnat_weights = np.array(lw)
        fw.close()

        with open(self.lcnat_src_vocab_path, 'r', encoding='utf-8') as fin:
            data_src = fin.readlines()
        self.lcnat_src_vocab = [line.strip() for line in data_src if len(line.strip())>0]

        with open(self.lcnat_tgt_vocab_path, 'r', encoding='utf-8') as fin:
            data_tgt = fin.readlines()
        self.lcnat_tgt_vocab = [line.strip() for line in data_tgt if len(line.strip())>0]

        assert len(task.source_dictionary) == self.lcnat_weights.shape[0] \
            and self.lcnat_weights.shape[0] == len(self.lcnat_src_vocab)

        assert len(task.target_dictionary) == self.lcnat_weights.shape[1] \
            and self.lcnat_weights.shape[1] == len(self.lcnat_tgt_vocab)

        # Check the vocabulary
        for widx_s in range(len(task.source_dictionary)):
            assert task.source_dictionary.symbols[wdix_s] == self.lcnat_src_vocab[widx_s]
        
        for widx_t in range(len(task.target_dictionary)):
            assert task.target_dictionary.symbols[widx_t] == self.lcnat_tgt_vocab[widx_t]
        
        # Get the lcnat lambda
        self.total_update_num = 300000.0
        self.lcnat_lambda = round(np.log2(self.total_update_num / (2 * (trainer.get_num_updates() + 1))) / np.log2(self.total_update_num / 2), 3)

    @staticmethod
    def add_args(parser):
        super(
            LabelSmoothedDualImitationCriterion,
            LabelSmoothedDualImitationCriterion,
        ).add_args(parser)
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            "--lcnat-weight-path",
            type=str,
            help="lcnat weight path"
        )
        parser.add_argument(
            "--lcnat-src-vocab-path",
            type=str,
            help="lcnat src vocabulary path"
        )
        parser.add_argument(
            "--lcnat-tgt-vocab-path",
            type=str,
            help="lcnat tgt vocabulary path"
        )

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        target_weights = torch.from_numpy(self.lcnat_weights[targets.squeeze(-1).cpu()])
        # kd_loss = F.kl_div(input=logits, target=target_weights)

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")
                kd_loss = F.kl_div(input=logits, target=target_weights)
                kd_loss = kd_loss.sim(-1)
            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        loss = loss * (1. - self.lcnat_lambda) + kd_loss * self.lcnat_lambda
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "kd_loss": kd_loss,"factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss, kd_loss = [], [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]
            if outputs[obj].get("kd_loss", False):
                kd_loss += [_losses.get("kd_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "kd_loss": kd_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        kd_loss = utils.item(sum(log.get("kd_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "kd_loss", kd_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
