import argparse
import torch

TEST = False

TOTAL_CLIENTS = 100
CLIENTS = 10
ROUNDS = 50
LOCAL_EPOCHS = 10
LR = 0.1
PROX = 0
BETA = 1.0

PRUNING_INTERVAL = 10
PRUNING_BEGIN = 0.9
READJUSTMENT_RATIO = 0.5
CACHE_TEST_SET_GPU = False
FP16 = False
SPARSITY = 0.1
FINAL_SPARSITY = 0.9
SPARSITY_DISTRIBUTION = 'erk'

ALIGNED_DST = False
ANCHOR_SELECTION_RULE = 'random'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if ALIGNED_DST:
    OUTFILE = 'output_aligned_dst.log'
else:
    OUTFILE = 'output_plain_dst.log'

config_parser = argparse.ArgumentParser()
config_parser.add_argument('-o', '--outfile', default=OUTFILE, type=argparse.FileType('a', encoding='ascii'))
CONFIG_ARGS = config_parser.parse_args()