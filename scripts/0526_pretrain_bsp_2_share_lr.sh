begin=0
end=3
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0526_pretrain_bsp_2_share_lr
export branch=master
export phase=pretrain

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh

epochs=100
blr=1e-4 # 1.5e-4
batch_size=512 # 512*2*4=4096 
accum_iter=2 
warmup_epochs=5
export MASTER_PORT=25924
prefix="$HOME/.exp/0520_MAE_yjw/$exp_name"

for repeat in {0,1,2}; do
    for times in {0,1,}; do
        export exp_args="--phase $phase --accum_iter $accum_iter --blr $blr --batch_size $batch_size --warmup_epochs $warmup_epochs --epochs $epochs --norm_pix_loss "
        
        if test $times -gt 0
        then
            export run_name="times=$repeat,bsp=$times,warmup_epochs=$warmup_epochs,phase=pretrain,norm_pix_loss=1,blr=$blr,bs=$((batch_size * accum_iter))"
            bash base.sh --share_lr 1 --epochs_used 100 --bsp_resume $prefix/*times=$repeat,bsp=$(($times-1))*/checkpoint.pth
        else
            export run_name="times=$repeat,bsp=$times,warmup_epochs=$warmup_epochs,phase=pretrain,norm_pix_loss=1,blr=$blr,bs=$((batch_size * accum_iter))"
            bash base.sh --share_lr 1 --epochs_used 0
        fi
    done
done
wait

bash 0526_finetune_bsp_2_share_lr.sh
bash 0526_linprobe_bsp_2_share_lr.sh
bash 0525_linprobe_bsp_2_normpix=0.sh
bash zombie.sh
