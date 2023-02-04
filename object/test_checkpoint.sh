# ***************************************************Pre-Adaptation
# Source LCE Models
# Vanilla Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier vanilla_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla/uda/VISDA-C/T

# PASTA (Weak) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a3k2b025_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025/uda/VISDA-C/T

# PASTA (Strong) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a10k1b05_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05/uda/VISDA-C/T

# Source AdaFocal 0.5 Models
# Vanilla Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier vanilla_adafocal_g05_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g0.5/uda/VISDA-C/T

# PASTA (Weak) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a3k2b025_adafocal_g05_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g05/uda/VISDA-C/T

# PASTA (Strong) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a10k1b05_adafocal_g05_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g05/uda/VISDA-C/T

# Source AdaFocal 2 Models
# Vanilla Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier vanilla_adafocal_g2_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g2/uda/VISDA-C/T

# PASTA (Weak) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a3k2b025_adafocal_g2_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g2/uda/VISDA-C/T

# PASTA (Strong) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a10k1b05_adafocal_g2_source \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g2/uda/VISDA-C/T


# ***************************************************Post-Adaptation


# Source LCE Models
# Vanilla Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier vanilla_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adapt/uda/VISDA-C/TV \
    --src_only 0

# PASTA (Weak) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a3k2b025_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adapt/uda/VISDA-C/TV \
    --src_only 0

# PASTA (Strong) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a10k1b05_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adapt/uda/VISDA-C/TV \
    --src_only 0

# Source AdaFocal 0.5 Models
# Vanilla Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier vanilla_adafocal_g05_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g0.5_adapt/uda/VISDA-C/TV \
    --src_only 0

# PASTA (Weak) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a3k2b025_adafocal_g05_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g05_adapt/uda/VISDA-C/TV \
    --src_only 0

# PASTA (Strong) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a10k1b05_adafocal_g05_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g05_adapt/uda/VISDA-C/TV \
    --src_only 0

# Source AdaFocal 2 Models
# Vanilla Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier vanilla_adafocal_g2_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g2_adapt/uda/VISDA-C/TV \
    --src_only 0

# PASTA (Weak) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a3k2b025_adafocal_g2_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g2_adapt/uda/VISDA-C/TV \
    --src_only 0

# PASTA (Strong) Source
python test_checkpoint.py \
    --dset VISDA-C \
    --identifier pasta_a10k1b05_adafocal_g2_adapt \
    --ckpt_dir /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g2_adapt/uda/VISDA-C/TV \
    --src_only 0