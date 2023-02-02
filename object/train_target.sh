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