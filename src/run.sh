#!/bin/bash
#SBATCH --job-name=run1
#SBATCH --output=../deepwalk/run
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate AllSet

python train.py --dname=mimic3 --epochs=100 --cuda=1 --num_labels=25 --num_nodes=100 --num_labeled_data=500
python train.py --dname=cradle --epochs=100 --cuda=1 --num_labels=1 --num_nodes=200 --num_labeled_data=all

python train.py --dname=mimic3 --epochs=100 --cuda=1 --num_labels=25 --num_nodes=100 --num_labeled_data=500 --vanilla
python train.py --dname=cradle --epochs=100 --cuda=1 --num_labels=1 --num_nodes=200 --num_labeled_data=all --vanilla

#for reg_lambda in 5 10 50; do
#for model_lambda in 0.1 0.5 1 5; do
#for view_lr in 0.005 0.001 5e-4; do
#python train.py --dname=mimic3-with-single --epochs=150 --All_num_layers=2 --cuda=1 --LearnFeat --reg_lambda=${reg_lambda} --model_lambda=${model_lambda} --view_lr=${view_lr}
#done
#done
#done

#for view_lambda in 0.001; do
#  for view_lr in 1e-2 5e-3 1e-3; do
#    for lr in 1e-2 5e-3 1e-3; do
#      for model_lambda in 0.001; do
#        python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=0  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2
#      done
#    done
#  done
#done

# view_lr 1e-2 lr 1e-2  79+ 82+ 71+
# view_lr 1e-2 lr 5e-3  79+ 83+ 72+  v
# view_lr 1e-2 lr 1e-3  79+ 83+ 72+  v
# view_lr 5e-3 lr 1e-2  79+ 82+ 71+
# view_lr 5e-3 lr 5e-3  79+ 83+ 72+  v
# view_lr 5e-3 lr 1e-3  79+ 83+ 72+  v
# view_lr 1e-3 lr 1e-2  79+ 82+ 71+
# view_lr 1e-3 lr 5e-3  79+ 83+ 72+  v
# view_lr 1e-3 lr 1e-3

#view_lambda=0.001
#model_lambda=0.001
#view_lr=1e-2
#lr=1e-2
#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=0  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2
#
#view_lr=1e-2
#lr=1e-3
#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=0  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2
#
#view_lr=5e-3
#lr=1e-3
#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=0  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2
#
#view_lr=1e-3
#lr=1e-3
#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=0  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2

#view_lambda=0.001
#model_lambda=0.01
#view_lr=1e-2
#lr=1e-3

#lr=5e-3
#wd=1e-6
#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=0 --cuda=6  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2
#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=1 --cuda=6  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2
#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=6  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2

#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=3  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --vanilla --dropout=0.5

#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=3  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --dropout=0.5

#python train.py --dname=mimic3-with-single --epochs=500 --All_num_layers=2 --cuda=3 --lr=${lr} --method=CEGCN --dropout=0.5

#python train.py --dname=mimic3-with-single --epochs=500 --All_num_layers=2 --cuda=3 --lr=${lr} --method=CEGAT --dropout=0.5

#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=6 --lr=${lr} --method=HNHN --MLP_hidden=128

#python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=6 --lr=${lr} --method=HCHA --MLP_hidden=128 --dropout=0.5 --wd=${wd}



#for model_lambda in 0.001 0.01 0.1 1; do
#  python train.py --dname=mimic3-with-single --epochs=1000 --All_num_layers=2 --cuda=5  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2
#done

#for view_lambda in 1e-4 0.001 0.01 0.1; do
#  python train.py --dname=mimic3-with-single --epochs=800 --All_num_layers=2 --cuda=5  --view_lambda=${view_lambda} --view_alpha=0.5 --model_lambda=${model_lambda} --view_lr=${view_lr} --lr=${lr} --gamma=5 --use_grad --grad_type=cagrad --grad_para=0.2
#done



