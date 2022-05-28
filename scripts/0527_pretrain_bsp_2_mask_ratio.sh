begin=0
end=3
para=1
parallel_per_gpu=1

export nproc_per_node=$((end - begin + 1))
export exp_name=0527_MAE_bsp_2_mask_ratio
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
prefix="$HOME/.exp/0520_MAE_yjw/0525_MAE_bsp_2"

for repeat in {0,1,2}; do
    for mask_rate in {0.75,0.8,0.85,0.9,0.95}; do
        for times in {1,}; do
            export exp_args="--phase $phase --accum_iter $accum_iter --blr $blr --batch_size $batch_size --warmup_epochs $warmup_epochs --norm_pix_loss --epochs $epochs"
            
            if test $times -gt 0
            then
                export run_name="times=$repeat,bsp=$times,mask_rate=$mask_rate,warmup_epochs=$warmup_epochs,phase=pretrain,norm_pix_loss=1,blr=$blr,bs=$((batch_size * accum_iter))"
                bash base.sh --bsp_resume $prefix/*times=$repeat,bsp=$(($times-1))*/checkpoint.pth --mask_ratio $mask_rate
            else
                export run_name="times=$repeat,bsp=$times,mask_rate=$mask_rate,warmup_epochs=$warmup_epochs,phase=pretrain,norm_pix_loss=1,blr=$blr,bs=$((batch_size * accum_iter))"
                bash base.sh 
            fi
        done
    done
done
wait

bash 0527_finetune_bsp_2_mask_ratio.sh
bash 0527_linprobe_bsp_2_mask_ratio.sh
bash zombie.sh