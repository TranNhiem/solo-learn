python3 ../../../main_pretrain_edit.py \
    --dataset mulda_v1 \
<<<<<<< HEAD
    --backbone resnet50 \
    --data_dir /data/ \
=======
    --MASSL_new True \
    --backbone resnet50 \
    --data_dir /data1/1K_New \
>>>>>>> 5ca1b247415e44e8f9481824751e9b12695a18d7
    --train_dir train \
    --val_dir val \
    --max_epochs 300 \
    --gpus 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --accumulate_grad_batches 1 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 0.4 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --color_jitter_prob 0.8 \
    --gray_scale_prob 0.2 \
    --horizontal_flip_prob 0.5 \
    --gaussian_prob 1.0  \
    --solarization_prob 0.0  \
    --crop_size 224 \
    --min_scale 0.08 \
    --max_scale 1.0 \
    --rda_num_ops 2 \
    --rda_magnitude 9 \
    --ada_policy imagenet \
    --fda_policy imagenet \
    --num_crops_per_aug 1 1 1 \
<<<<<<< HEAD
    --name MASSL_newDesign_norm-resnet50-imagenet-300ep \
=======
    --name MASSL_newDesign-resnet50-imagenet-300ep \
>>>>>>> 5ca1b247415e44e8f9481824751e9b12695a18d7
    --entity mlbrl \
    --project solo_MASSL \
    --wandb \
    --save_checkpoint \
<<<<<<< HEAD
    --method massl \
=======
    --method massl_edit \
>>>>>>> 5ca1b247415e44e8f9481824751e9b12695a18d7
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    --checkpoint_dir /data1/solo_MASSL_ckpt \
    --checkpoint_frequency 10 
