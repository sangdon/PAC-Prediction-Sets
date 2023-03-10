import os, sys
import argparse
import warnings
import numpy as np
import math
import pickle
import types

import torch as tc

import util
import data
import model
import learning
import uncertainty

    
def main(args):

    ## init datasets
    print("## init datasets: %s"%(args.data.src))    
    ds = getattr(data, args.data.src)(args.data)
    print()

    ## init a model
    print("## init models: %s"%(args.model.base))
    if 'FNN' in args.model.base or 'Linear' in args.model.base:
        mdl = getattr(model, args.model.base)(n_in=args.data.dim[0], n_out=args.data.n_labels, path_pretrained=args.model.path_pretrained)    
    elif 'ResNet' in args.model.base:
        mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels, path_pretrained=args.model.path_pretrained)
    else:
        raise NotImplementedError
    if args.multi_gpus:
        assert(not args.cpu)
        mdl = tc.nn.DataParallel(mdl).cuda()
    print()

    ## learn the model
    l = learning.ClsLearner(mdl, args.train)
    if args.model.path_pretrained is None:
        print("## train...")
        l.train(ds.train, ds.val, ds.test)
    print("## test...")
    if args.train.skip_eval:
        print('skip evaluation')
    else:
        l.test(ds.test, ld_name=args.data.src, verbose=True)
    print()

    
    ## prediction set estimation
    if args.train_ps.method == 'pac_predset_CP':
        mdl_ps = model.PredSetCls(mdl)
        l = uncertainty.PredSetConstructor_CP(mdl_ps, args.train_ps)
    else:
        raise NotImplementedError
    l.train(ds.val)
    l.test(ds.test, ld_name=f'test datasets', verbose=True)

    
if __name__ == '__main__':
    
    ## init a parser
    parser = argparse.ArgumentParser(description='PAC Prediction Set')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--multi_gpus', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=100)
    parser.add_argument('--data.n_workers', type=int, default=8)
    parser.add_argument('--data.src', type=str, default='ImageNet')
    parser.add_argument('--data.n_labels', type=int, default=1000)
    parser.add_argument('--data.seed', type=lambda v: None if v=='None' else int(v), default=0)
    
    ## model args
    parser.add_argument('--model.base', type=str, default='ResNet152')
    parser.add_argument('--model.path_pretrained', type=str, default='pytorch')

    ## train args
    parser.add_argument('--train.rerun', action='store_true')
    parser.add_argument('--train.load_final', action='store_true')
    parser.add_argument('--train.skip_eval', action='store_true')
    parser.add_argument('--train.resume', type=str)
    parser.add_argument('--train.method', type=str, default='src')
    parser.add_argument('--train.optimizer', type=str, default='SGD')
    parser.add_argument('--train.n_epochs', type=int, default=100)
    parser.add_argument('--train.lr', type=float, default=0.1) 
    parser.add_argument('--train.momentum', type=float, default=0.9)
    parser.add_argument('--train.weight_decay', type=float, default=0.0)
    parser.add_argument('--train.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--train.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--train.val_period', type=int, default=1)

    ## uncertainty estimation args
    parser.add_argument('--train_ps.method', type=str, default='pac_predset_CP')
    parser.add_argument('--train_ps.rerun', action='store_true')
    parser.add_argument('--train_ps.load_final', action='store_true')
    parser.add_argument('--train_ps.verbose', type=bool, default=True)
    parser.add_argument('--train_ps.save_compact', action='store_true')

    ## parameters for pac_predset_BC
    parser.add_argument('--train_ps.binary_search', action='store_true')
    parser.add_argument('--train_ps.bnd_type', type=str, default='direct')

    # parameters for pac_predset_CP
    parser.add_argument('--train_ps.T_step', type=float, default=1e-7) 
    parser.add_argument('--train_ps.T_end', type=float, default=np.inf)
    parser.add_argument('--train_ps.eps_tol', type=float, default=1.25)

    parser.add_argument('--train_ps.n', type=int, default=25000)
    parser.add_argument('--train_ps.eps', type=float, default=0.01)
    parser.add_argument('--train_ps.delta', type=float, default=1e-5)
            
    args = parser.parse_args()
    args = util.to_tree_namespace(args)
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args = util.propagate_args(args, 'device')
    args = util.propagate_args(args, 'exp_name')
    args = util.propagate_args(args, 'snapshot_root')

    ## setup logger
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))    

    ## print args
    util.print_args(args)

    ## run
    main(args)

