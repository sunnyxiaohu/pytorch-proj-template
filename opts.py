import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Action Detection (Recognition) Networks")

# ========================= Dataset Configs =========================
parser.add_argument('dataset', type=str, choices=['ActivityNet', 'Thumos14'])
parser.add_argument('--class_num', type=int, default=0,
                    help='will reset lately according the dataset')

# ========================= Model Configs ==========================
parser.add_argument('--model', default='ACT', help='model name')

parser.add_argument('--loss', type=str, default='1*CrossEntropy', 
                    help='loss function configuration')

parser.add_argument("--margin", type=float, default=1.2, help='')
parser.add_argument("--re_rank", action='store_true', help='')
parser.add_argument("--random_erasing", action='store_true', help='')
parser.add_argument("--probability", type=float, default=0.5, help='')

# ========================= Learning Configs ==========================
parser.add_argument('--optimizer', default='SGD', choices=('SGD','ADAM','NADAM','RMSprop'),
                    help='optimizer to use (SGD | ADAM | NADAM | RMSprop)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--dampening', type=float, default=0, 
                    help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', 
                    help='SGD nesterov')
parser.add_argument('--beta1', type=float, default=0.9, 
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, 
                    help='ADAM beta2')
parser.add_argument('--amsgrad', action='store_true', 
                    help='ADAM amsgrad')  
parser.add_argument('--gamma', type=float, default=0.1, 
                    help='learning rate decay factor for step decay')                                      
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--epsilon', type=float, default=1e-8, 
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--decay_type', type=str, default='step', 
                    help='learning rate decay type')
parser.add_argument('--lr_decay', type=int, default=5, 
                    help='learning rate decay per N epochs')

# ========================= Monitor Configs ==========================
parser.add_argument('--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test_every', type=int, default=1, 
                    help='do test per every N epochs')
parser.add_argument('--save_models', action='store_true', 
                    help='save all intermediate models')
                    
# ========================= Runtime Configs ==========================
parser.add_argument('--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', type=str, default='',
                    help='file name to save')
parser.add_argument('--nGPU', default=1, type=int, 
                    help='number of gpus used')
parser.add_argument('--cpu', action='store_true', 
                    help='set this option to cpu device')
parser.add_argument('--test_only', action='store_true', 
                    help='set this option to test the model')
parser.add_argument('--evaluate_only', action='store_true', 
                    help='set this option to evaluate the model')
parser.add_argument("--resume", type=int, default=0, 
                    help='resume from specific checkpoint, latest(-1), pre_train(0)')
parser.add_argument('--pre_train', type=str, default='', 
                    help='pre-trained model directory')
parser.add_argument('--reset', action='store_true', 
                    help='reset the training')
parser.add_argument('--evaluate_results', type=str, default='results.json',
                    help='the filename to save the evaluate results')

args = parser.parse_args()

if args.dataset == 'ActivityNet':
    args.class_num = 201
elif args.dataset == 'Thumos14':
    args.class_num == 21




