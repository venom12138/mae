begin=0
end=1
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0521_MAE_cifar10_DDP
export branch=master
export phase=linprobe

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh
blr=0.1
batch_size=8192

export exp_args="--cls_token --phase $phase --blr $blr --batch_size $batch_size --epochs 90 --dist_eval "

export run_name="norm_pix_loss=0,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune '/cluster/home1/yjw/.exp/0520_MAE_yjw/0521_MAE_cifar10_DDP/Y5105_warmup_epochs=5,norm_pix_loss=0,phase=pretrain,blr=1.5e-4,bs=512/checkpoint-199.pth.tar'
