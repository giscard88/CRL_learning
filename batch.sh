#! /bin/bash


python train_rl.py --simlength 1000

for i in 1
do

for j in 1000
do 

python run_qs.py --search 3 --rejection 5 --start $j --test 10 --torch_seed $i
#python run_qs.py --search 5 --rejection 5 --start $j --test 10 --torch_seed $i
#python run_qs.py --search 7 --rejection 5 --start $j --test 10 --torch_seed $i
#python run_qs.py --search 10 --rejection 5 --start $j --test 10 --torch_seed $i

python run_rl.py --start $j --test 10 --torch_seed $i

done


done




