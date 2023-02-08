#!/bin/sh -l
#SBATCH -p short
#SBATCH --gres gpu:1
#SBATCH --constraint=2080_ti
#SBATCH --pty bash
#SBATCH -J focalLoss_job
#SBATCH -o focalLoss_job.log

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate shot
cd /nethome/becsedi3/develop/SHOT_Prithvi/SHOT/object

echo "== Running 18 jobs =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job01 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 1/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job02 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 2/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job03 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 3/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job04 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 4/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda --dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job05 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 5/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job06 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 6/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g0.5 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job07 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 7/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g0.5 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job08 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 8/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g05 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job09 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 9/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g05 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job10 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 10/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g05 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job11 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 11/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g05 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job12 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 12/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g2 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job13 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 13/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/vanilla_adafocal_g2 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job14 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 14/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g2 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job15 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 15/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a3k2b025_adafocal_g2 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job16 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 16/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g2 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job17 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 2.0

echo "== Progress: 17/18 jobs ran successfully =="

srun -u python -u image_target.py \
	--cls_par 0.3 \
	--da uda \
	--dset VISDA-C \
	--gpu_id 0 \
	--s 0 \
	--output_src /coc/scratch/prithvi/dg_for_da/recognition_sfda/shot/pasta_a10k1b05_adafocal_g2 \
	--output /srv/hoffman-lab/share4/becsedi3/results/SHOT/focal_loss/job18 \
	--net resnet101 \
	--lr 1e-3 \
	--loss focal_loss \
	--gamma 0.5

echo "== Progress: 18/18 jobs ran successfully =="
echo "== Jobs complete =="

