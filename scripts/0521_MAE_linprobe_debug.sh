begin=0
end=3
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0521_MAE_cifar10_debug
export branch=master
export phase=linprobe

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh
blr=0.1
batch_size=4096

export exp_args="--cls_token --phase $phase --blr $blr --batch_size $batch_size --epochs 90 --dist_eval "

export run_name="maecode,norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash base.sh --finetune '/cluster/home1/yjw/venom/others/mae/output_dir/0521_MAE_times=0,norm_pix_loss=1,,warmup_epochs=5,phase=pretrain,blr=1.5e-4,bs=512/checkpoint-199.pth' &
export run_name="maecode,norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=4,5,6,7 bash base.sh --finetune '/cluster/home1/yjw/venom/others/mae/output_dir/0521_MAE_times=1,norm_pix_loss=1,,warmup_epochs=5,phase=pretrain,blr=1.5e-4,bs=512/checkpoint-199.pth' &
wait
export run_name="maecode,norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash base.sh --finetune '/cluster/home1/yjw/venom/others/mae/output_dir/0521_MAE_times=3,norm_pix_loss=1,,warmup_epochs=5,phase=pretrain,blr=1.5e-4,bs=512/checkpoint-199.pth' &

