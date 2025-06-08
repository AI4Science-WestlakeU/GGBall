#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

# python ../main.py +experiment=ego_small_3edge dataset=ego_small loss=VAE
# python ../main.py +experiment=ego_small_hyp dataset=ego_small loss=VQVAE model.hyp_channels=128 model.latent_channels=128 model.euc_channels=0 loss.codebook_size=4096 loss.lambda_commitment_weight=0.1 loss.lambda_orthogonal_reg_weight=0.0 loss.lambda_l2=0.0
export PYTHONPATH="/butianci/HypeFlow:$PYTHONPATH"
export PYTHONPATH="/butianci/HypeFlow/src/models/hyperbolic_nn_plusplus:$PYTHONPATH"

EXPERIMENT=qm9
DATASET=qm9
LOSS=VQVAE
CODEBOOK_SIZE=512
HYP_CHANNELS=128
EUC_CHANNELS=0
DEBUG=false
RIEMANNIAN_OPTIMIZER=true
DECODER_METHOD=strong_mlp
USE_PE=True
USE_RESNET=True
USE_LAYERNORM=False
SAMPLE_CODEBOOK_TEMP=None
USE_KMEANS=True
INTEGRATE_METHOD=vt_prediction

TASK=flow
VAE_MODEL_PATH="'/butianci/HypeFlow/src/outputs/epoch=3239-batch_loss=0.00.ckpt'"
# FLOW_MODEL_PATH="'/data/wangchuanrui/runs_graph/comm20_AE_32code_hyp64_euc0_decode_strong_mlp-comm20/flow/2025-05-02/14-01-07/every_n_epochs/epoch=39299-step=39300.ckpt'"
FLOW_MODEL_PATH=None

TEST_ONLY=False

python ../train_flow.py \
  +experiment=${EXPERIMENT} \
  dataset=${DATASET} \
  loss=${LOSS} \
  loss.codebook_size=${CODEBOOK_SIZE} \
  model.hyp_channels=${HYP_CHANNELS} \
  model.latent_channels=${HYP_CHANNELS} \
  model.euc_channels=${EUC_CHANNELS} \
  loss.lambda_commitment_weight=0.25 \
  loss.lambda_orthogonal_reg_weight=0.0 \
  loss.lambda_vq_loss_weight=0.0 \
  loss.lambda_l2=0.01 \
  loss.lambda_hyp2node=10.0 \
  loss.lambda_hyp2edge=10.0 \
  loss.use_kmeans=${USE_KMEANS} \
  loss.use_riemannian_optimizer=${RIEMANNIAN_OPTIMIZER} \
  dataset.debug=${DEBUG} \
  model.decode_method=${DECODER_METHOD} \
  model.decode_method=${DECODER_METHOD} \
  model.transformer_encoder.use_pe=${USE_PE} \
  model.transformer_decoder.use_pe=${USE_PE} \
  model.use_resnet=${USE_RESNET} \
  model.use_layernorm=${USE_LAYERNORM} \
  model.sample_codebook_temp=${SAMPLE_CODEBOOK_TEMP} \
  general.task=${TASK} \
  flow_train.VAE_checkpoint=${VAE_MODEL_PATH} \
  flow_train.Flow_checkpoint=${FLOW_MODEL_PATH} \
  flow_train.test_only=${TEST_ONLY} \
  flow_train.integrate.method=${INTEGRATE_METHOD}