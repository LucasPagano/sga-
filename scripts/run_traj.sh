#!/usr/bin/env bash
python train.py \
  --dataset_name 'trajnet_stanford/image_plane' \
  --delim space \
  --loader_num_workers 1 \
  --obs_len 8 \
  --pred_len 12 \
  --skip 1 \
  --batch_size 32 \
  --num_iterations 10000 \
  --num_epochs 200 \
  --embedding_dim 16 \
  --num_layers 1 \
  --dropout 0 \
  --mlp_dim 64 \
  --batch_norm 0 \
  --encoder_h_dim_g 32 \
  --decoder_h_dim_g 32 \
  --noise_dim 8 \
  --noise_type gaussian \
  --noise_mix_type global \
  --clipping_threshold_g 1.5 \
  --g_learning_rate 1e-3 \
  --g_steps 1 \
  --pooling_type 'pool_net' \
  --pool_every_timestep 0 \
  --bottleneck_dim 32 \
  --neighborhood_size 2.0 \
  --grid_size 8 \
  --d_type 'local' \
  --encoder_h_dim_d 64\
  --d_learning_rate 1e-3 \
  --d_steps 2 \
  --clipping_threshold_d 0 \
  --l2_loss_weight 1 \
  --best_k 10 \
  --output_dir save \
  --print_every 500 \
  --checkpoint_every 100 \
  --checkpoint_name checkpoint_SDD \
  --checkpoint_start_from None \
  --restore_from_checkpoint 0 \
  --num_samples_check 5000 \
  --use_gpu 1 \
  --timing 0 \
  --gpu_num 0 \



