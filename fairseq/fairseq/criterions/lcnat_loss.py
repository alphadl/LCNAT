# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor

import numpy as np
import h5py

@register_criterion("lcnat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, lcnat_weight_path, lcnat_src_vocab_path, lcnat_tgt_vocab_path):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.lcnat_weight_path = lcnat_weight_path
        self.lcnat_src_vocab_path = lcnat_src_vocab_path
        self.lcnat_tgt_vocab_path = lcnat_tgt_vocab_path

        assert self.lcnat_weight_path is not None
        assert self.lcnat_src_vocab_path is not None
        assert self.lcnat_tgt_vocab_path is not None

        fw = h5py.File(self.lcnat_weight_path, 'r')
        lw = fw['weights']
        self.lcnat_weights = np.array(lw)
        fw.close()

        with open(self.lcnat_src_vocab_path, 'r', encoding='utf-8') as fin:
            data_src = fin.readlines()
        self.lcnat_src_vocab = [line.strip() for line in data_src if len(line.strip()) > 0]

        with open(self.lcnat_tgt_vocab_path, 'r', encoding='utf-8') as fin:
            data_tgt = fin.readlines()
        self.lcnat_tgt_vocab = [line.strip() for line in data_tgt if len(line.strip()) > 0]

        assert len(task.source_dictionary) == self.lcnat_weights.shape[0]
        assert len(task.source_dictionary) == len(self.lcnat_src_vocab)
        assert len(task.target_dictionary) == self.lcnat_weights.shape[1]
        assert len(task.target_dictionary) == len(self.lcnat_tgt_vocab)

        for widx_s in range(len(task.source_dictionary)):
            assert task.source_dictionary.symbols[widx_s] == self.lcnat_src_vocab[widx_s]
        for widx_t in range(len(task.target_dictionary)):
            assert task.target_dictionary.symbols[widx_t] == self.lcnat_tgt_vocab[widx_t]

        self.total_update_num = 300000.0

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing",
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
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0, num_updates=0
    ):
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0.0, device=outputs.device)
            kd_loss = torch.tensor(0.0, device=outputs.device)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")
                nll_loss = mean_ds(losses)
                if label_smoothing > 0:
                    loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                else:
                    loss = nll_loss

                # Data-dependent prior: KL(prior || model); lambda decays over first half of training
                lam = self._lambda(num_updates)
                if lam > 0:
                    target_weights = torch.from_numpy(
                        self.lcnat_weights[targets.cpu().numpy()]
                    ).to(logits.device).float()
                    target_weights = target_weights.clamp(min=1e-8)
                    target_weights = target_weights / target_weights.sum(dim=-1, keepdim=True)
                    kd_loss = F.kl_div(logits, target_weights, reduction="batchmean")
                    loss = loss * (1.0 - lam) + kd_loss * lam
                else:
                    kd_loss = torch.tensor(0.0, device=logits.device)
            else:
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none").sum(-1)
                nll_loss = mean_ds(losses)
                if label_smoothing > 0:
                    loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                else:
                    loss = nll_loss
                kd_loss = torch.tensor(0.0, device=logits.device)

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "kd_loss": kd_loss, "factor": factor}

    def _lambda(self, num_updates):
        """Decay: lambda = log(I/(2(i+1)))/log(I/2) for i <= I/2 else 0."""
        I = self.total_update_num
        if num_updates >= I // 2:
            return 0.0
        return max(0.0, np.log2(I / (2 * (num_updates + 1))) / np.log2(I / 2))

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {
            "name": name, "loss": loss, "factor": factor,
            "nll_loss": loss.new_tensor(0.0), "kd_loss": loss.new_tensor(0.0),
        }

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
        num_updates = model.get_num_updates() if hasattr(model, "get_num_updates") else 0

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                    num_updates=num_updates,
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
                else l["loss"].data / l["factor"]
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
