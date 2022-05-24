begin=0
end=1
para=1
parallel_per_gpu=1

export nproc_per_node=2 # $((end - begin + 1))
export exp_name=0521_MAE_cifar10_debug
export branch=master
export phase=linprobe

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh
blr=0.1
batch_size=8192

export exp_args="--cls_token --phase $phase --blr $blr --batch_size $batch_size --epochs 90 --dist_eval "
export MASTER_PORT=25789
export run_name="maecode,norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=0,1 bash base.sh --finetune '/home/jwyu/venom/mae/output_dir/0521_MAE_times=0,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth.tar' &
export MASTER_PORT=25755
export run_name="maecode,norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=2,3 bash base.sh --finetune '/home/jwyu/venom/mae/output_dir/0521_MAE_times=1,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth.tar' &
wait
export MASTER_PORT=25789
export run_name="maecode,norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=0,1 bash base.sh --finetune '/home/jwyu/venom/mae/output_dir/0521_MAE_times=2,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth.tar' &

export exp_name=0521_MAE_test_var
export MASTER_PORT=25755
export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=0,1 bash base.sh --finetune '/home/jwyu/.exp/0520_MAE_yjw/0521_MAE_test_var/Y0005_warmup_epochs=5,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth' &
wait

export MASTER_PORT=25755
export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=0,1 bash base.sh --finetune '/home/jwyu/.exp/0520_MAE_yjw/0521_MAE_test_var/Y0006_warmup_epochs=5,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth' &
export MASTER_PORT=25783
export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
CUDA_VISIBLE_DEVICES=0,1 bash base.sh --finetune '/home/jwyu/.exp/0520_MAE_yjw/0521_MAE_test_var/Y0007_warmup_epochs=5,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth' &
