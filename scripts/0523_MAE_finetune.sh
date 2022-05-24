begin=0
end=3
para=1
parallel_per_gpu=1
. utils.sh

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES
export MASTER_PORT=25789
export nproc_per_node=$((end - begin + 1))
export exp_name=0521_MAE_cifar10_debug
export branch=master

export phase=finetune

blr=5e-4
batch_size=256 # 256*4=1024

export exp_args="--phase $phase --blr $blr --batch_size $batch_size --epochs 100 --dist_eval "

export run_name="maecode,normpix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune '/home/jwyu/venom/mae/output_dir/0521_MAE_times=0,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth.tar'
export run_name="maecode,normpix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune '/home/jwyu/venom/mae/output_dir/0521_MAE_times=1,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth.tar'
export run_name="maecode,normpix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune '/home/jwyu/venom/mae/output_dir/0521_MAE_times=2,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth.tar'

export exp_name=0521_MAE_test_var

export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune '/home/jwyu/.exp/0520_MAE_yjw/0521_MAE_test_var/Y0005_warmup_epochs=5,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth' 
export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune '/home/jwyu/.exp/0520_MAE_yjw/0521_MAE_test_var/Y0006_warmup_epochs=5,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth' 
export run_name="norm_pix_loss=1,phase=$phase,blr=$blr,bs=$batch_size"
bash base.sh --finetune '/home/jwyu/.exp/0520_MAE_yjw/0521_MAE_test_var/Y0007_warmup_epochs=5,norm_pix_loss=1,phase=pretrain,blr=1.5e-4,bs=1024/checkpoint-199.pth' 