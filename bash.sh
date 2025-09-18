


# CG-FKAN
python3 ./train.py --model kan --method FedAvg --local-learning-rate 0.001 --gpu 3 --comm-rounds 1 --sparsification 1 --grid_varing --grid 3 --problem_id 11 --seed 1


# FKAN with grid extension
python3 ./train.py --model kan --method FedAvg --local-learning-rate 0.001 --gpu 3 --comm-rounds 1 --sparsification 0 --grid_varing --grid 3 --problem_id 11 --seed 1



# grid-fixed FKAN 
python3 ./train.py --model kan --method FedAvg --local-learning-rate 0.001 --gpu 3 --comm-rounds 1 --sparsification 0 --grid 3 --problem_id 11 --seed 1


# MLP with large weights
python3 ./train.py --model mlp --method FedAvg --local-learning-rate 0.001 --gpu 3 --comm-rounds 1 --sparsification 0 --grid 200 --problem_id 11 --seed 1