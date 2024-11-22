# Data

With Xview2 Data preprocessed as so:

data/

- full/
  - post-disaster/
    - images/
    - targets/
  - pre-disaster/
    - images/
    - targets/

run the `make_patches script`. Modify the script for the various patch sizes desired, the default is just 256.

*Note*: Running `data.py` will show you what the data looks like.

# Running the Model

Install the prerequisites with `conda env create -f environment.yml`, then run `run.py`

# Architecture

- `damage_extractor` -- as of now unused script used make the dataset on disk have an even number of damaged and undamaged samples.
  For now that's handled in code.

- `full_dataset` -- This file contains all the code relating to the dataset, augmentations, and dataloaders.

- `data` -- All code responsible for datasets and dataloaders.

- `learner` -- Classes responsible for the training of a model system and collecting metrics.

- `models` -- Model systems comprised of a model, dataloaders, data transforms, and a loss function, since these all depend on each other.

- `run` -- the actual training of the model. Hyperparameter optimization may be added back in again here.

- `torch_utils` -- miscellaneous pytorch utilities of varying usefullness.

# Possibility of Change Detection

https://github.com/likyoo/open-cd
