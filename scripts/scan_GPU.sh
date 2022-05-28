echo $1
while :
do
	STRING0=`nvidia-smi -a -i 0|grep Free|head -1| tr -cd "[0-9]"`
	STRING1=`nvidia-smi -a -i 1|grep Free|head -1| tr -cd "[0-9]"`
	STRING2=`nvidia-smi -a -i 2|grep Free|head -1| tr -cd "[0-9]"`
	STRING3=`nvidia-smi -a -i 3|grep Free|head -1| tr -cd "[0-9]"`
	B=18000
    echo $STRING0
	if [[ "$STRING0" -lt "$B" ]] || [[ "$STRING1" -lt "$B" ]] || [[ "$STRING2" -lt "$B" ]] || [[ "$STRING3" -lt "$B" ]]; then
		echo `date -d today +"%Y-%m-%d %H:%M:%S"`:GPU 0:$STRING0
		sleep 30
	else
		if [ $? -eq 0 ]; then
			curl "https://api.day.app/YvWQ6m97tqedNPHGqNjZPP/Finished/($exp_name)"
		else
			curl "https://api.day.app/YvWQ6m97tqedNPHGqNjZPP/Failed/($exp_name)"
		fi
		bash zombie.sh
        break
	fi
done
