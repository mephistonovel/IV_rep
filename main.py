import argparse
from Experiment.experiment import experiment_Syn,experiment_Real
import warnings
from numba.core.errors import (NumbaDeprecationWarning, 
                                    NumbaPendingDeprecationWarning,
                                    NumbaPerformanceWarning)
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.filterwarnings('ignore')
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="IV_rep")
    parser.add_argument("--feature-dim", default=6, type=int)
    parser.add_argument("--latent-dim", default=2, type=int)
    parser.add_argument("--latent-dim-t", default=3, type=int)
    parser.add_argument("--latent-dim-y", default=1, type=int)
    parser.add_argument("--hidden-dim", default=200, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument("--num-epochs", default=128, type=int)
    parser.add_argument("--sample_size", type=int, default=5000,help="Sample size")
    parser.add_argument("--batch_size", type=int, default=512,help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float,help='learning rate')
    parser.add_argument("--lrd",  default=0.01, type=float,help="learning-rate-decay")
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--treatment", type=str,default='con',help="Treatment type - b: binary, cat: categorical, con: continuous")
    parser.add_argument("--comp-num-zc", type=int,default=2,help="number of components of iv(zt)")
    parser.add_argument("--model_id",default = 'autoiv', type=str, help = 'model type')
    parser.add_argument("--hyp_reconst", type=float,default=1,help="hyperparam of weight for reconstruction term p(d|z,c)")
    parser.add_argument("--hyp_ztloss", type=float,default=1,help="hyperparam of weight for loss related to zt(z)")
    parser.add_argument("--hyp_zcloss", type=float,default=1,help="hyperparam of weight for loss related to zc(c)")
    parser.add_argument("--hyp_tdist", type=float,default=5,help="hyperparam of weight for treatment dist p(t|zt)")
    parser.add_argument("--hyp_treg", type=float,default=5,help="hyperparam of weight for treatment regression f(t|zt,zc)")
    parser.add_argument("--hyp_yreg", type=float,default=5,help="hyperparam of weight for outcome regression f(y|t,zc)")
    parser.add_argument("--hyp_mi", type=float,default=5,help="hyperparam of weight for MI loss (zt,zc)")
    parser.add_argument('--pretrain', type=bool, default=True,help='GMM pretrain')
    parser.add_argument('--pretrained_path', type=str, default='./weights/pretrained_parameter.pth',
                        help='Output path')
    parser.add_argument('--GPU_id', type=str, default='0', help='gpu id to execute code when gpu is true')
    parser.add_argument('--exp_id', type=str, default='syn', help='experiment id')
    parser.add_argument('--repetition', type=int, default=20, help='repetition')
    parser.add_argument('--dependency', type=str, default='nodep', help='within d1, d2 dependency')
    parser.add_argument('--interaction', type=str, default='nointer', help='d1 d2 data interaction')
    parser.add_argument('--response', type=str, default='linear', help='form of response function')
    parser.add_argument('--highdim', type=int, default=0, help='highdim')
    parser.add_argument('--true_effect', type=int, default=3, help='true effect')
    parser.add_argument('--use_dist_net', type=bool, default=True,help='construct p(x|z)')
    parser.add_argument('--use_reg_net', type=bool, default=True,help='construct f(y|x^,c),f(x|z,c)')
    parser.add_argument('--use_reconst_x', type=bool, default=True,help='x^ for f(y|x^,c) vs x')    
    parser.add_argument('--use_flex_enc', type=bool, default=True,help='q(c|d) VaDE vs VAE')

    
    args = parser.parse_args()
    if args.exp_id == 'syn':
        experiment_Syn(args, args.repetition, args.sample_size)

  