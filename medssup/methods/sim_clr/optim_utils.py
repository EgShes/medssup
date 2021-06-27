import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_iter(
    model: nn.Module,
    batch: Dict[str, Any],
    optimizer: Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()

    with autocast(enabled=scaler.is_enabled()):
        pos = model(batch["pos"])
        neg = model(batch["neg"])
        loss = criterion(pos, neg)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


@torch.no_grad()
def eval_iter(
    model: nn.Module, batch: Dict[str, Any], criterion: nn.Module, scaler: GradScaler, device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}

    with autocast(enabled=scaler.is_enabled()):
        pos = model(batch["pos"])
        neg = model(batch["neg"])
        loss, logits, labels = criterion(pos, neg, return_logits=True)

    acc = accuracy(logits, labels, topk=(1, 5))

    return loss.item(), acc[0], acc[1]


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, scaler: GradScaler, device: torch.device
) -> Dict[str, float]:
    metrics = defaultdict(list)
    for batch in tqdm(loader, desc="Evaluating", total=len(loader)):
        loss, acc1, acc5 = eval_iter(model, batch, criterion, scaler, device)
        metrics["loss"].append(loss)
        metrics["acc1"].append(acc1)
        metrics["acc5"].append(acc5)
    return {name: np.mean(values) for name, values in metrics.items()}


class SimCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        self._similarity = nn.CosineSimilarity()

    def forward(
        self, pos_logits: Tensor, neg_logits: Tensor, return_logits: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        pos_logits = F.normalize(pos_logits, dim=1)
        neg_logits = F.normalize(neg_logits, dim=1)

        # similarity between positives
        # for example we have vector of positives [1, 2, 3] and want to compare each other with the next sample
        # so we do [1, 2, 3] + [1] => [1, 2, 3, 1] => [2, 3, 1]
        #             [1, 2,    3] =>    [1, 2, 3] => [1, 2, 3]

        pos_sim = self._similarity(torch.cat([pos_logits, pos_logits[:1, :]], dim=0)[1:], pos_logits)

        # similarity between positives and negatives
        # each positive is compared with each negative

        pos_neg_sim = torch.matmul(pos_logits, neg_logits.T)

        logits = torch.cat([pos_sim.unsqueeze(1), pos_neg_sim], dim=1)
        # maximize elements on the 0 position in each row -> similarity between positives
        labels = torch.zeros(logits.shape[0]).long().to(logits.device)

        loss = super().forward(logits, labels)

        if return_logits:
            return loss, logits, labels

        return loss


@torch.no_grad()
def accuracy(output: Tensor, target: Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def load_optimizer(model: nn.Module, optimizer_type: str, learning_rate: float, epochs: int, weight_decay: float = 0.0):
    if optimizer_type == "adam":
        optimizer_type = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "lars":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        # learning_rate = 0.3 * batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)
    else:
        raise NotImplementedError(f"Wrong optimizer type. Must be adam or lars. Got {optimizer_type}")

    return optimizer, scheduler


EETA_DEFAULT = 0.001


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/lars.py
class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            # epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            # weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            # eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    # trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.ge(0),
                        torch.where(g_norm.ge(0), (self.eeta * w_norm / g_norm), torch.Tensor([1.0]).to(device)),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True
