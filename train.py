import torch
import argparse

from utils import *
from server import *
from dataset import *
from model import FastKANClassifier, MLPClassifier
from feynman import get_feynman_dataset
from kan.MultKAN import KAN
import math

print("##=============================================##")
print("##     Federated Learning Simulator Starts     ##")
print("##=============================================##")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help="gpu number")
parser.add_argument('--dataset', type=str, default='feynman')
parser.add_argument('--problem_id', type=int, default=1)
parser.add_argument('--model', type=str, default='ekan')                
parser.add_argument('--non-iid', action='store_true', default=False)                                       
parser.add_argument('--split-rule',type=str, default='Quantity_skew')  
parser.add_argument('--split-coef', default=100.0, type=float)                                                  
parser.add_argument('--active-ratio', default=0.1, type=float)                                             
parser.add_argument('--total-client', default=100, type=int)                                              
parser.add_argument('--comm-rounds', default=1200, type=int)                                               
parser.add_argument('--num_class', default=8, type=int)
parser.add_argument('--local-epochs', default=5, type=int)                                                
parser.add_argument('--batchsize', default=512, type=int)                                                 
parser.add_argument('--weight-decay', default=0.001, type=float)                                         
parser.add_argument('--local-learning-rate', default=0.1, type=float)                                   
parser.add_argument('--lr-decay', default=0.998, type=float)                                         
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--layer', default='1', type=str)                                                
parser.add_argument('--grid', default=10, type=int)
parser.add_argument('--spline_order', default=3, type=int)

parser.add_argument('--data-file', default='./', type=str)                                              
parser.add_argument('--out-file', default='out/', type=str)
# select the path of the log files
parser.add_argument('--save-model', action='store_true', default=False)
parser.add_argument('--use-RI', action='store_true', default=False)
parser.add_argument('--sparsification', default=1, type=int)
parser.add_argument('--sparse-ratio', default=0.0, type=float)
parser.add_argument('--sparse_method', type=str, default='TopK', help="method option:fixed/random/TopK")
parser.add_argument('--grouping_method', type=str, default='none', help="method option:portion/layer")
parser.add_argument('--quantization_levels', default=0.0, type=float)

parser.add_argument('--debugging_log', type=int, default=0, help="1 represents debugging log, 0 represents normal log")
parser.add_argument('--grid_varing', action='store_true', help="True: represents varying grid, False: represents normal log")
parser.add_argument('--sparse_varing', action='store_true', help="True: represents varying sparsification, False: fixed sparsification")
parser.add_argument('--act_grid', action='store_true', help="True: only coef, sparsification")
parser.add_argument('--restricted_grid_size', type=int, default=10, help="10")
parser.add_argument('--restricted_bits', type=int, default=1e12, help="1e12")


parser.add_argument('--sparse_alpha', default=0.1, type=float, help="Top-k, shared")
parser.add_argument('--sparse_beta', default=1, type=float, help="Layer-wise sparsification ratio")
parser.add_argument('--sparse_gamma', default=1e-2, type=float, help="Stop layer-wise sparsification ratio based on training RMSE")

parser.add_argument('--mlp_increasing', action='store_true')

parser.add_argument('--alpha', default=0.1, type=float)                                                    
parser.add_argument('--beta', default=0.1, type=float)                                                     
parser.add_argument('--beta1', default=0.9, type=float)                                                    
parser.add_argument('--beta2', default=0.99, type=float)                                                   
parser.add_argument('--lamb', default=0.1, type=float)                                                     
parser.add_argument('--rho', default=0.001, type=float)                                                   
parser.add_argument('--gamma', default=1.0, type=float)                                                    
parser.add_argument('--epsilon', default=0.01, type=float)                                                 
parser.add_argument('--method', choices=['FedAvg', 'FedCM', 'FedDyn', 'SCAFFOLD', 'FedAdam', 'FedProx', 'FedSAM', 'MoFedSAM', \
                                         'FedGamma', 'FedSpeed', 'FedSMOO'], type=str, default='FedAvg')

args = parser.parse_args()

hidden_configs = {
    "p30": [5, 5],
    "p11": [5],
    "p38": [3,2],
    "p128": [2],
    "p124": [2, 2],
    "test": [1]
}


torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.out_file=f'seed{args.seed}'
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

if __name__=='__main__':
    ### Generate IID or Heterogeneous Dataset
    if not args.non_iid:
        data_obj = DatasetObject(n_client=args.total_client, seed=args.seed, unbalanced_sgm=0, rule='iid', data_path=args.data_file, args=args)
        print("Initialize the Dataset     --->  {:s} {:s} {:d} clients".format(args.dataset, 'IID', args.total_client))
    else:
        print(args.split_rule, args.split_coef)
        data_obj = DatasetObject(n_client=args.total_client, seed=args.seed, unbalanced_sgm=0, rule=args.split_rule, rule_arg=args.split_coef, data_path=args.data_file, args=args)
        print("Initialize the Dataset     --->  {:s} {:s}-{:s} {:d} clients".format(args.dataset, args.split_rule, str(args.split_coef), args.total_client))

    input_vars, expr, f, ranges = get_feynman_dataset(args.problem_id)

    if args.problem_id == 11:
        args.layer = "p11"
    elif args.problem_id == 30:
        args.layer = "p30"
    elif args.problem_id == 38:
        args.layer = "p38"
    elif args.problem_id == 124:
        args.layer = "p124"
    elif args.problem_id == 128:
        args.layer = "p128"
    elif args.problem_id == 1:
        args.layer = "test"
    else:
        raise NotImplementedError(f"Problem ID {args.problem_id} is not implemented yet.")
    print(f"Using layer configuration: {args.layer} with hidden layers: {hidden_configs[args.layer]}")

    ### Generate Model Function
    if args.model == 'mlp':
        print(f"hidden_configs[args.layer]: {hidden_configs[args.layer]}")
        target_weights = calculate_kan_weights(args.spline_order, int(args.grid), hidden_configs[args.layer], len(input_vars))
        input_dim = len(input_vars)
        args.sparsification = 0
        num_kan_hidden_layers = len(hidden_configs[args.layer])
        hidden_layers = calculate_mlp_multi_hidden_layers(target_weights, input_dim, 1, num_kan_hidden_layers)
        if args.mlp_increasing is True:
            hidden_layers = [25, 25]
        model_func = lambda: MLPClassifier(input_shape   = (len(input_vars),1,1), hidden_layers=hidden_layers, num_classes   = 1)
        model = model_func()

    elif args.model == 'kan':
        layers_hidden = []
        layers_hidden.append(len(input_vars))
        for i in range(len(hidden_configs[args.layer])):
            layers_hidden.append(hidden_configs[args.layer][i])
        layers_hidden.append(1)  
        model_func = lambda: KAN(width=layers_hidden, grid=args.grid, k=args.spline_order)
        model = model_func().to(args.device)

        model_func_temp = lambda: KAN(width=layers_hidden, grid=args.restricted_grid_size, k=args.spline_order)
        model_temp = model_func_temp().to(args.device)
        restricted_bits = 0
        for name, param in model_temp.named_parameters():
            print(f"{name}: {param.numel()}")
            restricted_bits += param.numel()
        args.restricted_bits = math.ceil(restricted_bits*float_bits)
        print(f"Total number of parameters in the restricted model: {restricted_bits}, which is approximately {args.restricted_bits} bits.")

    else:
        model_func = lambda: FastKANClassifier(num_classes=args.num_class, kan_hidden=hidden_configs[args.layer], num_grids=args.grid)
    print("Initialize the Model Func  --->  {:s} model".format(args.model))
    init_model = model_func()
    # init_model.to(args.device)
    total_trainable_params = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    print("                           --->  {:d} parameters".format(total_trainable_params))
    init_par_list = get_mdl_params(init_model)

    ### Generate Server
    server_func = None
    server_func = FedAvg

    _server = server_func(device=args.device, model_func=model_func, init_model=init_model, init_par_list=init_par_list,
                          datasets=data_obj, method=args.method, args=args)
    _server.train()

