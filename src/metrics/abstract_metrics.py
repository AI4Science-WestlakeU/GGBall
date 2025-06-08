import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError
from typing import Optional

class TrainAbstractMetricsDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class TrainAbstractMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMSE(MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
            """ Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
            tensors.
                preds: Predicted tensor
                target: Ground truth tensor
            """
            diff = preds - target
            sum_squared_error = torch.sum(diff * diff)
            n_obs = preds.shape[0]
            return sum_squared_error, n_obs


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.weight = weight

    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None, weight: Optional[Tensor] = None) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        target=target.view(-1)
        preds=preds.view(target.shape[0], -1)
        
        if weight is not None:  
            weight = weight.to(preds.device)
        if mask is not None:
            mask = mask.view(-1)
            valid_indices = mask > 0
            target = target[valid_indices]
            preds = preds[valid_indices]
        
        output = F.cross_entropy(preds, target, reduction='sum', weight=weight)
        self.total_ce += output
        self.total_samples += preds.size(0)

        # self.total_ce = output
        # self.total_samples = 0.0 * self.total_samples + max(preds.shape[0], 1)

    def compute(self):
        return self.total_ce / self.total_samples
    

class BCELossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_bce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        if mask is not None:
            mask = mask.view(-1)
            valid_indices = mask > 0
            target = target[valid_indices]
            preds = preds[valid_indices]
        output = F.binary_cross_entropy(preds, target, reduction='sum')
        self.total_bce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_bce / self.total_samples

class MSELossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_mse', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        if mask is not None:
            mask = mask.view(-1)
            valid_indices = mask > 0
            target = target[valid_indices]
            preds = preds[valid_indices]
        output = F.mse_loss(preds, target, reduction='sum')
        self.total_mse += output
        self.total_samples += preds.size(0)
    
    def compute(self):
        return self.total_mse / self.total_samples
        

class ProbabilityMetric(Metric):
    def __init__(self):
        """ This metric is used to track the marginal predicted probability of a class during training. """
        super().__init__()
        self.add_state('prob', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        self.prob += preds.sum()
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples
    
class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor = None) -> None:
        """Update state with predictions and targets."""
        target = torch.argmax(target, dim=-1)  # Convert target to class indices
        preds = torch.softmax(preds, dim=-1)       # Convert logits to probabilities
        preds = torch.multinomial(preds.view(-1, preds.size(-1)), 1).view(preds.size()[:-1])
        
        # Apply mask if provided
        if mask is not None:
            preds = preds[mask]
            target = target[mask]

        # Accumulate correct predictions and total samples
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self):
        """Compute accuracy."""
        return self.correct / self.total
    
class F1ScoreMetric(Metric):
    def __init__(self, average: str = 'micro'):
        """
        Initialize the F1ScoreMetric.
        Args:
            average: One of 'micro', 'macro', or 'weighted'. Default is 'micro'.
        """
        super().__init__()
        self.average = average
        self.add_state('true_positives', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('false_positives', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('false_negatives', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Tensor=None) -> None:
        """Update state with predictions and targets."""
        target = torch.argmax(target, dim=-1)  # Convert target to class indices
        preds = torch.softmax(preds, dim=-1)       # Convert logits to probabilities
        preds = torch.multinomial(preds.view(-1, preds.size(-1)), 1).view(preds.size()[:-1])
        
        # preds = torch.argmax(preds, dim=-1)    # Get predicted class indices
        
        # Apply mask if provided
        if mask is not None:
            preds = preds[mask]
            target = target[mask]

        # Calculate TP, FP, FN
        for cls in torch.unique(target):
            cls = cls.item()
            true_positive = ((preds == cls) & (target == cls)).sum().float()
            false_positive = ((preds == cls) & (target != cls)).sum().float()
            false_negative = ((preds != cls) & (target == cls)).sum().float()
            
            self.true_positives += true_positive
            self.false_positives += false_positive
            self.false_negatives += false_negative

    def compute(self):
        """Compute F1-Score."""
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if self.average == 'micro':
            return f1.mean()  # Micro-average: Global precision and recall
        elif self.average == 'macro':
            return f1.mean()  # Macro-average: Average over all classes
        elif self.average == 'weighted':
            return (f1 * self.true_positives).sum() / self.true_positives.sum()
        else:
            raise ValueError(f"Unsupported average type: {self.average}")