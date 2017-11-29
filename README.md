# Weight Reset for Distributional Shift

See [this post](http://jalexvig.github.io/blog/dropout-weight-reset/) for more details.

### Example use

```
python3 main.py dropout_replace
python3 main.py dropout_no-replace --prob_keep_perm 1
python3 main.py no-dropout_replace --prob_keep 1
python3 main.py no-dropout_no-replace --prob_keep 1 --prob_keep_perm 1
```

### Options

```
usage: main.py [-h] [--seed SEED] [--num_layers NUM_LAYERS]
               [--num_units [NUM_UNITS [NUM_UNITS ...]]]
               [--activation_funcs [{tanh,relu} [{tanh,relu} ...]]]
               [--prob_keep [PROB_KEEP [PROB_KEEP ...]]]
               [--learning_rate LEARNING_RATE]
               [--num_data_perturbations NUM_DATA_PERTURBATIONS]
               [--num_epochs NUM_EPOCHS] [--loss_cutoff LOSS_CUTOFF]
               [--prob_keep_perm [PROB_KEEP_PERM [PROB_KEEP_PERM ...]]]
               [--quiet]
               exp_name

positional arguments:
  exp_name

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED, -s SEED
  --num_layers NUM_LAYERS, -nl NUM_LAYERS
  --num_units [NUM_UNITS [NUM_UNITS ...]], -nu [NUM_UNITS [NUM_UNITS ...]]
  --activation_funcs [{tanh,relu} [{tanh,relu} ...]], -af [{tanh,relu} [{tanh,relu} ...]]
  --prob_keep [PROB_KEEP [PROB_KEEP ...]], -pk [PROB_KEEP [PROB_KEEP ...]]
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --num_data_perturbations NUM_DATA_PERTURBATIONS, -ndp NUM_DATA_PERTURBATIONS
  --num_epochs NUM_EPOCHS, -ne NUM_EPOCHS
  --loss_cutoff LOSS_CUTOFF, -lc LOSS_CUTOFF
  --prob_keep_perm [PROB_KEEP_PERM [PROB_KEEP_PERM ...]], -pkp [PROB_KEEP_PERM [PROB_KEEP_PERM ...]]
  --quiet, -q
```