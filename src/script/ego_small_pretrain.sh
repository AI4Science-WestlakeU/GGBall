#! /bin/bash
source /opt/conda/bin/activate
conda activate hypeflow
export PYTHONPATH="/butianci/HypeFlow:$PYTHONPATH"
export PYTHONPATH="/butianci/HypeFlow/src/models/hyperbolic_nn_plusplus:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0
# python ../main.py +experiment=ego_small_3edge dataset=ego_small loss=VAE
# python ../main.py +experiment=ego_small_hyp dataset=ego_small loss=VQVAE model.hyp_channels=128 model.latent_channels=128 model.euc_channels=0 loss.codebook_size=4096 loss.lambda_commitment_weight=0.1 loss.lambda_orthogonal_reg_weight=0.0 loss.lambda_l2=0.0
EXPERIMENT=ego_small_hyp
DATASET=ego_small
LOSS=AE
CODEBOOK_SIZE=32
HYP_CHANNELS=64
EUC_CHANNELS=0
DEBUG=false
RIEMANNIAN_OPTIMIZER=True

python ../main.py \
  +experiment=${EXPERIMENT} \
  dataset=${DATASET} \
  loss=${LOSS} \
  loss.codebook_size=${CODEBOOK_SIZE} \
  model.hyp_channels=${HYP_CHANNELS} \
  model.latent_channels=${HYP_CHANNELS} \
  model.euc_channels=${EUC_CHANNELS} \
  loss.lambda_commitment_weight=1.0 \
  loss.lambda_orthogonal_reg_weight=0.1 \
  loss.lambda_vq_loss_weight=0.25 \
  loss.use_riemannian_optimizer=${RIEMANNIAN_OPTIMIZER} \
  dataset.debug=${DEBUG} \
  loss.lambda_l2=0.01 \
  loss.lambda_hyp2node=10 \
  loss.lambda_hyp2edge=10