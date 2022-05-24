echo $1
while :
do
	STRING0=`nvidia-smi -a -i 0|grep Free|head -1| tr -cd "[0-9]"`
	STRING1=`nvidia-smi -a -i 1|grep Free|head -1| tr -cd "[0-9]"`
	STRING2=`nvidia-smi -a -i 2|grep Free|head -1| tr -cd "[0-9]"`
	STRING3=`nvidia-smi -a -i 3|grep Free|head -1| tr -cd "[0-9]"`
	B=18000
    echo $STRING0
	if [[ "$STRING0" -lt "$B" ]] && [[ "$STRING1" -lt "$B" ]] && [[ "$STRING2" -lt "$B" ]] && [[ "$STRING3" -lt "$B" ]]; then
		echo `date -d today +"%Y-%m-%d %H:%M:%S"`:GPU 0:$STRING0
		sleep 30
	else
		# `CUDA_VISIBLE_DEVICES=$1 ./darknet detector train cfg/voc_tiny.data cfg/yolov3-tiny.cfg yolov3-tiny.conv.15 -dont_show -mjpeg_port 8090 -map`
        break
	fi
done
