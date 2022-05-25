begin=0
end=3
para=1
parallel_per_gpu=1
. utils.sh

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES
export MASTER_PORT=25789
export nproc_per_node=$((end - begin + 1))
export exp_name=0525_MAE_bsp_3
export branch=master

export phase=finetune

blr=5e-4
batch_size=256 # 256*4=1024
accum_iter=1

prefix="/home/jwyu/.exp/0520_MAE_yjw/$exp_name"
export exp_args="--phase $phase --blr $blr --batch_size $batch_size --epochs 100 --dist_eval "

export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune $prefix/*times=0,bsp=2,warmup_epochs=5,norm_pix_loss=1,phase=pretrain,*/checkpoint.pth

export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune $prefix/*times=1,bsp=2,warmup_epochs=5,norm_pix_loss=1,phase=pretrain,*/checkpoint.pth

export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune $prefix/*times=2,bsp=2,warmup_epochs=5,norm_pix_loss=1,phase=pretrain,*/checkpoint.pth