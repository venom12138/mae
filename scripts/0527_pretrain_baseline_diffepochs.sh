begin=0
end=3
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0527_MAE_bsp_2_tune_epochs
export branch=master
export phase=pretrain

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh

# epochs=100
blr=1e-4 # 1.5e-4
batch_size=512 # 512*2*4=4096
accum_iter=2 
warmup_epochs=5
export MASTER_PORT=25924
prefix="$HOME/.exp/0520_MAE_yjw/$exp_name"

for repeat in {0,1,2}; do
    for epochs0 in {100,125,150,175}; do
        for times in {0,1,}; do
            export exp_args="--phase $phase --accum_iter $accum_iter --blr $blr --batch_size $batch_size --warmup_epochs $warmup_epochs --norm_pix_loss "
            if test $times -gt 0
            then
                export run_name="times=$repeat,epochs=$((200-epochs0)),bsp=$times,warmup_epochs=$warmup_epochs,phase=pretrain,norm_pix_loss=1,blr=$blr,bs=$((batch_size * accum_iter))"
                bash base.sh --epochs $((200-epochs0)) --bsp_resume $prefix/*times=$repeat,epochs=$epochs0,bsp=$(($times-1))*/checkpoint.pth
            else
                export run_name="times=$repeat,epochs=$epochs0,bsp=$times,warmup_epochs=$warmup_epochs,phase=pretrain,norm_pix_loss=1,blr=$blr,bs=$((batch_size * accum_iter))"
                bash base.sh --epochs $epochs0
            fi
        done
    done
done
wait

bash 0527_finetune_bsp_2_tune_epochs.sh
bash 0527_linprobe_bsp_2_tune_epochs.sh
bash zombie.sh
