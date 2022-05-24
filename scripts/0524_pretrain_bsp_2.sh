begin=0
end=3
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0524_bsp_debug
export branch=master
export phase=pretrain

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh

epochs=100
blr=6.25e-5 # 1.5e-4
batch_size=512 # 512*2*4=4096 
accum_iter=2 
warmup_epochs=5
export MASTER_PORT=25924
prefix="$HOME/.exp/0520_MAE_yjw/$exp_name"

for times in {0,1,}; do
    export exp_args="--phase $phase --accum_iter $accum_iter --norm_pix_loss --blr $blr --batch_size $batch_size --warmup_epochs $warmup_epochs --epochs $epochs "
    export run_name="bsp=$times,warmup_epochs=$warmup_epochs,norm_pix_loss=1,phase=pretrain,blr=$blr,bs=$((batch_size * accum_iter))"
    
    if test $times -gt 0
    then
        bash base.sh --bsp_resume $prefix/*bsp=$(($times-1))*/checkpoint.pth
    else
        bash base.sh
    fi
done
wait

bash 0524_linprobe_bsp_2.sh
bash 0524_finetune_bsp_2.sh
bash zombie.sh
