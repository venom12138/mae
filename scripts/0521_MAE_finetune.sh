begin=0
end=7
para=1
parallel_per_gpu=1
. utils.sh

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES
export nproc_per_node=$((end - begin + 1))
export exp_name=0521_MAE_cifar10_DDP
export branch=master


export phase=finetune

blr=5e-4
batch_size=128 # 8*128=1024

export exp_args="--phase $phase --blr $blr --batch_size $batch_size --epochs 100 --dist_eval "

export run_name="phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune '/cluster/home1/yjw/.exp/0520_MAE_yjw/0521_MAE_cifar10_DDP/Y5100_warmup_epochs=5,phase=pretrain,blr=1.5e-4,bs=512/checkpoint-199.pth.tar'
