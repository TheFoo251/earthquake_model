# # Research Goal: test as many different loss functions as possible for THIS SPECIFIC USE!

# [https://walkwithfastai.com/Segmentation]


import torch
from fastai.data.all import *
from fastai.vision.all import *
from patchify import patchify
from PIL import Image
import optuna
from optuna.integration import FastAIPruningCallback
import optuna.visualization.matplotlib as vs
#import monai.losses as mdlss #(med loss)
from extra_loss_functions import *
import logging
import sys


# NON-OPTIMIZED HYPERPARAMS (cause my GPU can't handle it :P)

# the gpu can hold 256 * 256 * 8 pixels = 524288
#  or 128 * 128 * 32
# or 64 * 64 * 128
# or 32 * 32 * 512
SIZES = [(256, 8), (128, 32), (64, 128), (32, 512)]
PATCH_SZ, BATCH_SZ = SIZES[0]

# possibility of increasing bs/ps using fp16??? what about other modes??


# Set these low for testing hyper-optimizer setup. May be hyperparams later.
FREEZE_EPOCHS = 2
EPOCHS = 8
NUM_TRIALS = 150




# ### Pre-pipeline processing with patchify


# ### Make Optimizer


patch_dir = Path(f"../data/{PATCH_SZ}_patches")
path = patch_dir
codes = ["Background", "NoDamage", "MinorDamage", "MajorDamage", "Destroyed"]

dls = SegmentationDataLoaders.from_label_func(path, bs=BATCH_SZ,
    fnames = get_image_files(path/"images"), 
    label_func = lambda o: path/"targets"/f"{o.stem}{o.suffix}",                                     
    codes = codes,
    # batch_tfms=[*aug_transforms(size=(360,480)), Normalize.from_stats(*imagenet_stats)]
    )


# Don't add unimportant things. The number of trials increases exponentially with each added variable.
def objective(trial):

    #things the optimizer does...

    #pretrained = trial.suggest_categorical("pretrained", [True, False])



    
    #value for tversky ( change each of these to categorical...)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 0.0, 1.0)

    loss_dict = {
            "cross_entropy":CrossEntropyLossFlat(axis=1),
            "dice":DiceLoss(reduction="mean"),
            "tversky":ModifiedTverskyLoss(reduction="mean", beta=beta),
            "combo":ComboLoss(alpha=alpha),
            "focal_dice":FocalDiceLoss(alpha=alpha),
            "focal_tversky":FocalTverskyLoss(reduction="mean", alpha=alpha, beta=beta),
            "log_cosh_dice":LogCoshDiceLoss(reduction="mean"),
            }
    
    loss_fn = trial.suggest_categorical("loss_fn",
                                        [
                                        "cross_entropy",
                                        "dice",
                                        "tversky",
                                        "combo",
                                        "focal_dice",
                                        "focal_tversky",
                                        "log_cosh_dice",
                                        ]
                                       )
                                       
    
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2)

    

    learn = unet_learner(
        dls, 
        resnet18, 
        metrics=DiceMulti(axis=1),
        self_attention=True, 
        act_cls=Mish,
        loss_func = loss_dict[loss_fn],
        pretrained=True,
        n_out = len(codes) # set codes implicitly later
    )
    learn.to_fp16()

    model_cbs = [
    # EarlyStoppingCallback(monitor='valid_loss', min_delta=0.1, patience=2), # detect overfitting
    # EarlyStoppingCallback(monitor='train_loss', min_delta=0.1, patience=3), # decect stalled training
    # ActivationStats(with_hist=True)], # too slow
    FastAIPruningCallback(trial, monitor="dice_multi")
    # set this to `train_loss` to purposely overfit?
    # ! Optimizer may lose information on overfitting I need to look at... make sure to log everything.
    # TRY USING FP.16!!
    ]

    # See https://forums.fast.ai/t/how-to-diable-progress-bar-completely/65249/3
    # to disable progress bar and logging info.
    with learn.no_bar():
        with learn.no_logging():
            learn.fine_tune(epochs=EPOCHS,
                    base_lr=lr,
                    freeze_epochs=FREEZE_EPOCHS,
                    cbs=model_cbs
                   )

    return learn.recorder.metrics[0].value.item() # only one metric to worry about


# ### Optimize




# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "fastai-study-1"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)


study.optimize(objective, n_trials=NUM_TRIALS, timeout=None)


# print a bunch of junk

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

