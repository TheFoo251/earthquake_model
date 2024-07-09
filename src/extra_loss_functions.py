import torch
from fastai.data.all import *
from fastai.vision.all import *

__all__ = ["ModifiedTverskyLoss", "ComboLoss", "FocalDiceLoss", "FocalTverskyLoss", "LogCoshDiceLoss"]

class ModifiedTverskyLoss:
    """
    Modified Tversky Loss
    proposed in https://arxiv.org/abs/1706.05721
    based on FastAI's dice loss implementation
    Beta of 0.5 should equal dice loss exactly,
    but fastai's inbuilt dice is preferred because it requires less computation.
    Modified to be analytically equal to dice, even with smoothing
    """
    def __init__(self, 
        axis:int=1, # Class axis
        smooth:float=1e-6, # Helps with numerical stabilities in the IoU division
        reduction:str="sum", # PyTorch reduction to apply to the output
        beta = 0.5 # makes dice by default...
    ):
        store_attr()
        
    def __call__(self, pred:Tensor, targ:Tensor) -> Tensor:
        """One-hot encodes targ, then takes tversky index"""
        targ = self._one_hot(targ, pred.shape[self.axis])
        pred, targ = TensorBase(pred), TensorBase(targ)
        assert pred.shape == targ.shape, 'input and target dimensions differ, Tversky expects non one-hot targs'
        pred = self.activation(pred)
        sum_dims = list(range(2, len(pred.shape)))
        #set operations
        inter = torch.sum(pred*targ, dim=sum_dims)
        pred_not_in_targ = torch.sum((1-targ)*pred, dim=sum_dims) # relative complements
        targ_not_in_pred = torch.sum((1-pred)*targ, dim=sum_dims)
        # calculate index
        num = inter
        denom = inter + self.beta*pred_not_in_targ + (1-self.beta)*targ_not_in_pred #TODO check for reversing...
        tversky_index = (num + (self.smooth/2))/(denom + (self.smooth/2))
        #get loss
        loss = 1 - tversky_index
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    @staticmethod
    def _one_hot(
        x:Tensor, # Non one-hot encoded targs
        classes:int, # The number of classes 
        axis:int=1 # The axis to stack for encoding (class dimension)
    ) -> Tensor:
        """Creates one binary mask per class"""
        return torch.stack([torch.where(x==c, 1, 0) for c in range(classes)], axis=axis)
    
    def activation(self, x:Tensor) -> Tensor: 
        """Activation function applied to model output"""
        return F.softmax(x, dim=self.axis)
    
    def decodes(self, x:Tensor) -> Tensor:
        """Converts model output to target format"""
        return x.argmax(dim=self.axis)

class ComboLoss: 
    """
    Dice and Cross Entropy Loss combined
    Proposed in [SOURCE]
    """
    def __init__(self, axis=1, smooth=1., alpha=1., reduction="sum"):
        store_attr()
        self.ce_loss = CrossEntropyLossFlat(axis=axis)
        self.dice_loss =  DiceLoss(axis, smooth, reduction=reduction)
        
    def __call__(self, pred, targ):
        return self.alpha * self.ce_loss(pred, targ) + (1-self.alpha)*self.dice_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

class FocalDiceLoss: # example class building on dice loss
    """
    Dice and Focal Loss combined
    proposed in [SOURCE]
    """
    def __init__(self, axis=1, smooth=1., alpha=1., reduction="sum"):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=self.axis)
        self.dice_loss =  DiceLoss(axis, smooth, reduction=reduction)
        
    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

class FocalTverskyLoss: # TODO-- use the one in the paper!
    """
    Tversky and Focal Loss combined
    proposed in [SOURCE]
    Fundementally different from FocalDiceLoss?? (we shall see)
    """
    def __init__(self, axis=1, smooth=1., alpha=1., beta=0.5, reduction="sum"):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.tversky_loss =  ModifiedTverskyLoss(axis, smooth, reduction=reduction, beta=beta) #TODO-- fix param passing
        
    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.tversky_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

class LogCoshDiceLoss:
    """
    proposed in http://arxiv.org/abs/2006.14822
    """
    def __init__(self, axis=1, smooth=1., reduction="sum"):
        store_attr()
        self.dice_loss =  DiceLoss(axis, smooth, reduction=reduction)
        
    def __call__(self, pred, targ):
        x = self.dice_loss(pred, targ)
        return torch.log(torch.cosh(x))
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)