# Vanilla Model (L)CE
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss label_smooth_ce

python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss label_smooth_ce

# Vanilla Model AdaFocal
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g2 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 2.0

python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g2_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 2.0

# Vanilla Model AdaFocal
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g0.5 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 0.5

python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g0.5_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 0.5

# PASTA (Weak) Model (L)CE
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss label_smooth_ce \
    --use_pasta 1 \
    --pasta_a 3.0 \
    --pasta_k 2.0 \
    --pasta_b 0.25


# PASTA (Weak) Model AdaFocal
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g2 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 2.0 \
    --use_pasta 1 \
    --pasta_a 3.0 \
    --pasta_k 2.0 \
    --pasta_b 0.25

python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g2_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 2.0 \
    --use_pasta 1 \
    --pasta_a 3.0 \
    --pasta_k 2.0 \
    --pasta_b 0.25

# PASTA (Weak) Model AdaFocal
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g05 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 0.5 \
    --use_pasta 1 \
    --pasta_a 3.0 \
    --pasta_k 2.0 \
    --pasta_b 0.25

python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g05_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 0.5 \
    --use_pasta 1 \
    --pasta_a 3.0 \
    --pasta_k 2.0 \
    --pasta_b 0.25

##############

# PASTA (Weak) Model (L)CE
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss label_smooth_ce \
    --use_pasta 1 \
    --pasta_a 10.0 \
    --pasta_k 1.0 \
    --pasta_b 0.5


# PASTA (Weak) Model AdaFocal
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g2 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 2.0 \
    --use_pasta 1 \
    --pasta_a 10.0 \
    --pasta_k 1.0 \
    --pasta_b 0.5

python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g2_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 2.0 \
    --use_pasta 1 \
    --pasta_a 10.0 \
    --pasta_k 1.0 \
    --pasta_b 0.5

# PASTA (Weak) Model AdaFocal
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g05 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 10 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 0.5 \
    --use_pasta 1 \
    --pasta_a 10.0 \
    --pasta_k 1.0 \
    --pasta_b 0.5

python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g05_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss adaptive_focal_loss \
    --gamma 0.5 \
    --use_pasta 1 \
    --pasta_a 10.0 \
    --pasta_k 1.0 \
    --pasta_b 0.5

# Train for longer
# PASTA (Weak) Model (L)CE
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss label_smooth_ce \
    --use_pasta 1 \
    --pasta_a 3.0 \
    --pasta_k 2.0 \
    --pasta_b 0.25


# PASTA (Strong) Model (L)CE
python image_source.py \
    --trte val \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_ep30 \
    --da uda \
    --gpu_id 0 \
    --dset VISDA-C \
    --net resnet101 \
    --lr 1e-3 \
    --max_epoch 30 \
    --s 0 \
    --loss label_smooth_ce \
    --use_pasta 1 \
    --pasta_a 10.0 \
    --pasta_k 1.0 \
    --pasta_b 0.5