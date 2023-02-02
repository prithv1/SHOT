# Vanilla Model (L)CE + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adapt \
    --net resnet101 \
    --lr 1e-3

# Vanilla Model AdaFocal + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g2 \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g2_adapt \
    --net resnet101 \
    --lr 1e-3

# Vanilla Model AdaFocal + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g0.5 \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g0.5_adapt \
    --net resnet101 \
    --lr 1e-3

# PASTA
# PASTA (Weak) Model (L)CE + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025 \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adapt \
    --net resnet101 \
    --lr 1e-3

# PASTA (Weak) Model AdaFocal + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g2 \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g2_adapt \
    --net resnet101 \
    --lr 1e-3

# PASTA (Weak) Model AdaFocal + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g05 \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g05_adapt \
    --net resnet101 \
    --lr 1e-3

# PASTA
# PASTA (Strong) Model (L)CE + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05 \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adapt \
    --net resnet101 \
    --lr 1e-3

# PASTA (Strong) Model AdaFocal + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g2 \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g2_adapt \
    --net resnet101 \
    --lr 1e-3

# PASTA (Strong) Model AdaFocal + Vanilla Adapt
python image_target.py \
    --cls_par 0.3 \
    --da uda \
    --dset VISDA-C \
    --gpu_id 0 \
    --s 0 \
    --output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g05 \
    --output /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g05_adapt \
    --net resnet101 \
    --lr 1e-3