#! /bin/bash
source /opt/conda/bin/activate
conda activate hypeflow
export PYTHONPATH="/butianci/HypeFlow:$PYTHONPATH"
export PYTHONPATH="/butianci/HypeFlow/src/models/hyperbolic_nn_plusplus:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1
# python ../main.py +experiment=ego_small_3edge dataset=ego_small loss=VAE
# codebook = 32, 128, 512, 2048
# hyp_dim = 8, 16, 32, 64, 128
EXPERIMENT=comm20_hyp
DATASET=comm20
LOSS=VQVAE
CODEBOOK_SIZE=32
HYP_CHANNELS=64
EUC_CHANNELS=0
DEBUG=false
RIEMANNIAN_OPTIMIZER=True
USE_RESNET=True
USE_LAYER_NORM=False
SAMPLE_CODEBOOK_TEMP=None

python ../main.py \
  +experiment=${EXPERIMENT} \
  dataset=${DATASET} \
  loss=${LOSS} \
  loss.codebook_size=${CODEBOOK_SIZE} \
  model.hyp_channels=${HYP_CHANNELS} \
  model.latent_channels=${HYP_CHANNELS} \
  model.euc_channels=${EUC_CHANNELS} \
  loss.lambda_commitment_weight=1 \
  loss.lambda_orthogonal_reg_weight=0.1 \
  loss.lambda_vq_loss_weight=0.25 \
  loss.use_riemannian_optimizer=${RIEMANNIAN_OPTIMIZER} \
  dataset.debug=${DEBUG} \
  model.use_resnet=${USE_RESNET} \
  model.use_layernorm=${USE_LAYER_NORM} \
  model.sample_codebook_temp=${SAMPLE_CODEBOOK_TEMP}