#! /bin/bash

python train_rl.py --simlength 5000

for i in 1 2 3 4 5 6 7 8 9 10
do

for j in 1000 2000 3000 4000 5000
do 

python run_qs.py --search 3 --rejection 5 --start $j --test 100 --torch_seed $i
python run_qs.py --search 5 --rejection 5 --start $j --test 100 --torch_seed $i
python run_qs.py --search 7 --rejection 5 --start $j --test 100 --torch_seed $i
python run_qs.py --search 10 --rejection 5 --start $j --test 100 --torch_seed $i

python run_rl.py --start $j --test 100 --torch_seed $i

done


done




