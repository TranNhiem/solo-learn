python3 ../../../main_pretrain_edit.py \
    --dataset mulda_v1 \
    --backbone resnet50 \
<<<<<<< HEAD
    --data_dir /data/ \
=======
    --data_dir /data1/1K_New/ \
>>>>>>> 0c0dc9838e18af5b3fa0da10889a0ed83db441f6
    --train_dir train \
    --val_dir val \
    --max_epochs 600 \
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
    --lr 0.5 \
    --accumulate_grad_batches 2 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 0.4 0.4 0.4\
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --color_jitter_prob 0.8 \
    --gray_scale_prob 0.2 \
    --horizontal_flip_prob 0.5 \
    --gaussian_prob 1.0  \
    --solarization_prob 0.2  \
    --crop_size 224 \
    --min_scale 0.08 \
    --max_scale 1.0 \
    --rda_num_ops 2 \
    --rda_magnitude 9 \
    --ada_policy imagenet \
    --fda_policy imagenet \
<<<<<<< HEAD
    --num_crops_per_aug 1 1 1 \
    --name MASSL_newDesign_3MLP-resnet50-imagenet-300ep \
=======
    --num_crops_per_aug 1 1 1 1  \
    --name MASSL_newDesign-resnet50-imagenet-600ep \
>>>>>>> 0c0dc9838e18af5b3fa0da10889a0ed83db441f6
    --entity mlbrl \
    --project solo_MASSL \
    --wandb \
    --save_checkpoint \
    --method massl_edit \
    --proj_output_dim 512 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    --checkpoint_dir /data1/solo_MASSL_ckpt \
    --checkpoint_frequency 10 
