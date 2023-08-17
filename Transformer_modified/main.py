import argparse

import torch


def train(args, model):

    train_loader, test_loader = None #get_loader(args)
    
    # TODO: cases for the optimizer arg
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    # t_total = arg
    pass

def main():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./ckpts')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--ff_hidden_size', type=int, default=512)
    parser.add_argument('--num_head', type=int, default=4)
    parser.add_argument('--encoder_seq_length', type=int, default=20)
    parser.add_argument('--decoder_seq_length', type=int, default=20)
    parser.add_argument('--total_step', type=int, default=100000)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
