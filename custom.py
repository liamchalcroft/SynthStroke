import torch
from monai.transforms.transform import MapTransform
from monai.utils.enums import TransformBackends
import monai as mn
from typing import (
    Callable,
    Dict,
    Hashable,
    Mapping,
    Optional,
)
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Randomizable
from monai.transforms import KeepLargestConnectedComponent
import cornucopia as cc
import warnings


class RandomSkullStrip(MapTransform, Randomizable):
    """ """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        label_key="label",
        image_key="image",
        out_key="mask",
        channels_to_use=[0, 1, 2, 3, 9],
        dilate_prob=0.3,
        erode_prob=0.3,
    ) -> None:
        MapTransform.__init__(self, [label_key], allow_missing_keys=False)
        self.label_key = label_key
        self.image_keys = [image_key] if not isinstance(image_key, (list, tuple)) else image_key
        self.out_key = out_key
        self.channels_to_use = channels_to_use
        self.fill = mn.transforms.FillHoles()
        self.dilate = cc.DilateLabelTransform(radius=2)
        self.r_dilate = cc.RandomDilateLabelTransform(labels=dilate_prob, radius=2)
        self.r_erode = cc.RandomErodeLabelTransform(labels=erode_prob, radius=4)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        mask = (d[self.label_key][self.channels_to_use] > 0.5).sum(axis=0).int()[None]
        mask = KeepLargestConnectedComponent(num_components=1)(mask)
        for i in range(3):
            mask = self.dilate(mask)
            mask = self.fill(mask)
        mask = self.r_dilate(mask)
        mask = self.r_erode(mask)
        for image_key in self.image_keys:
            d[image_key] = mask * d[image_key]
        del mask
        return d


class RemapSegToLabel(MapTransform, Randomizable):
    """ """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        in_key="seg",
        out_key="label",
    ) -> None:
        MapTransform.__init__(self, [in_key], allow_missing_keys=False)
        self.in_key = in_key
        self.out_key = out_key

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        d[self.out_key] = d[self.in_key]
        if self.in_key + "_meta_dict" in list(d.keys()):
            d[self.out_key+'_meta_dict'] = d[self.in_key+'_meta_dict']

        return d
    

class DiceCEL2Loss(torch.nn.modules.loss._Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        l2_epochs=None,
        l2_target=5,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        reduction = mn.utils.look_up_option(reduction, mn.utils.DiceCEReduction).value
        self.dice = mn.losses.DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not mn.utils.pytorch_after(1, 10)
        self.l2_epochs = l2_epochs
        self.l2_loss = torch.nn.MSELoss(reduction=reduction)
        self.epoch = 0
        if not l2_epochs:
            self.l2_epochs = -1
        self.l2_target = l2_target


    def l2(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
            target = torch.nn.functional.one_hot(target, n_pred_ch).transpose(0,1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        weights = (1 - target[:,0] + 1e-8).unsqueeze(1)
        return torch.sum(weights * (input - self.l2_target * (2 * target - 1))**2) / (torch.sum(weights) * n_pred_ch)

        return self.l2_loss(input, target)


    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)
        if self.cross_entropy.weight is not None:
            if not self.cross_entropy.weight.device == target.device:
                self.cross_entropy.weight = self.cross_entropy.weight.to(target.device)

        return self.cross_entropy(input, target)


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )     
        if input.shape[1] != target.shape[1] and target.shape[1] == 2:
            input = input[:,[0,-1]] # extract background + lesion
        if self.epoch < self.l2_epochs:
            total_loss: torch.Tensor = self.l2(input, target)
        else:
            dice_loss = self.dice(input, target)
            ce_loss = self.ce(input, target)
            total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss
