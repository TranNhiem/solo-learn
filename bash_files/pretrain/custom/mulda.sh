# Train without labels.
# To train with labels, simply remove --no_labels
# --val_dir is optional and will expect a directory with subfolder (classes)
# --dali flag is also supported

# you are free to modified the args described follows
python3 ../../../main_pretrain.py \
    --dataset mulda \
    --backbone resnet50 \
    --data_dir /data1/1K_New \
    --train_dir train \
    --no_labels val \
    --max_epochs 300 \
    --gpus 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 128 \
    --num_workers 4 \

    ## SimCLR DA options  ( ref : solo.args.utils.additional_setup_pretrain
    --brightness 0.4 0.4 0.4 0.4 \  # 4 params to bypass inner chk mechnaism
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --color_jitter_prob 0.8 \
    --gray_scale_prob 0.2 \
    --horizontal_flip_prob 0.5 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \

    ## multiple-DA options
    -- \
    -- \
    -- \

    # 4 view with diff trfs
    --num_crops_per_aug 1 1 1 1 \

    --name byol-400ep-custom \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --method byol \
    --output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 8192 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0
