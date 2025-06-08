#! /bin/bash
CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/butianci/HypeFlow:$PYTHONPATH"
export PYTHONPATH="/butianci/HypeFlow/src/models/hyperbolic_nn_plusplus:$PYTHONPATH"

CODEBOOK_SIZE=32
HYP_CHANNELS=64
EUC_CHANNELS=0


python ../train_flow.py +experiment=ego_small_hyp dataset=ego_small loss=VQVAE \
  model.hyp_channels=${HYP_CHANNELS} \
  model.latent_channels=${HYP_CHANNELS} \
  model.euc_channels=${EUC_CHANNELS} \
  loss.codebook_size=${CODEBOOK_SIZE}