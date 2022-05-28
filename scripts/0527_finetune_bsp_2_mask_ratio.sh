begin=0
end=3
para=1
parallel_per_gpu=1
. utils.sh

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES
export MASTER_PORT=25789
export nproc_per_node=$((end - begin + 1))
export exp_name=0527_MAE_bsp_2_mask_ratio
export branch=master

export phase=finetune

blr=5e-4
batch_size=256 # 256*4=1024
accum_iter=1

prefix="/home/jwyu/.exp/0520_MAE_yjw/$exp_name"
export exp_args="--phase $phase --blr $blr --batch_size $batch_size --epochs 100 --dist_eval "
for mask_rate in {0.75,0.8,0.85,0.9,0.95}; do
    export run_name="mask_rate=$mask_rate,phase=$phase,blr=$blr,bs=$batch_size"
    bash base.sh --finetune $prefix/*times=0,bsp=1,mask_rate=$mask_rate,warmup_epochs=5,phase=pretrain,*/checkpoint.pth

    export run_name="mask_rate=$mask_rate,phase=$phase,blr=$blr,bs=$batch_size"
    bash base.sh --finetune $prefix/*times=1,bsp=1,mask_rate=$mask_rate,warmup_epochs=5,phase=pretrain,*/checkpoint.pth 

    export run_name="mask_rate=$mask_rate,phase=$phase,blr=$blr,bs=$batch_size"
    bash base.sh --finetune $prefix/*times=2,bsp=1,mask_rate=$mask_rate,warmup_epochs=5,phase=pretrain,*/checkpoint.pth 
done