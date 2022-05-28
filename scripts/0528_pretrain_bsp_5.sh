begin=0
end=3
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0528_MAE_bsp_5
export branch=master
export phase=pretrain

export CUDA_VISIBLE_DEVICES="$(seq -s "," $begin $end)"
echo $CUDA_VISIBLE_DEVICES

. utils.sh

epochs=40
blr=1e-4 # 1.5e-4
batch_size=512 # 512*2*4=4096 
accum_iter=2 
warmup_epochs=5
export MASTER_PORT=25924
prefix="$HOME/.exp/0520_MAE_yjw/$exp_name"
for repeat in {0,1,2}; do
    for times in {3,4}; do
        export exp_args="--phase $phase --accum_iter $accum_iter --norm_pix_loss --blr $blr --batch_size $batch_size --warmup_epochs $warmup_epochs --epochs $epochs"
        export run_name="times=$repeat,bsp=$times,warmup_epochs=$warmup_epochs,norm_pix_loss=1,phase=pretrain,blr=$blr,bs=$((batch_size * accum_iter))"
        
        if test $times -gt 0
        then
            bash base.sh --bsp_resume $prefix/*times=$repeat,bsp=$(($times-1))*/checkpoint.pth
        else
            bash base.sh
        fi
    done
done
wait

bash 0528_linprobe_bsp_5.sh
bash 0528_finetune_bsp_5.sh

