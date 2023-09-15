import os
import sys
import logging
import argparse
import random
import math
import json
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils import redirect_stdout
from config import Config
from dataset import MyDataset
from models.transformer import Transformer

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def valid(model, ds_iter, training_config, checkpoint_path, global_step, best_dev_rmse, init_t, criterion):
    val_rmse = []
    val_mae = []
    eval_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for dev_step_idx in range(training_config["num_eval_steps"]):
            _, batch = next(ds_iter['dev'])

            batch['user_seq'] = batch['user_seq'].cuda()
            batch['user_degree'] = batch['user_degree'].cuda()
            batch['item_list'] = batch['item_list'].cuda()
            batch['item_degree'] = batch['item_degree'].cuda()
            batch['item_rating'] = batch['item_rating'].cuda()
            batch['spd_matrix'] = batch['spd_matrix'].cuda()

            outputs = model(batch)

            loss = criterion(outputs.float(), batch['item_rating'].float())
            # eval_losses.update(loss.mean())
            eval_losses.update(loss)

            # Compute RMSE & MAE
            mse = F.mse_loss(outputs, batch['item_rating'], reduction='none')
            rmse = torch.sqrt(mse.mean())
            mae = F.l1_loss(outputs, batch['item_rating'], reduction='mean')

            val_rmse.append(rmse)
            val_mae.append(mae)

        total_rmse = sum(val_rmse) / len(val_rmse)
        total_mae = sum(val_mae) / len(val_mae)
        if total_rmse > best_dev_rmse:
            best_dev_accu = total_rmse
            torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
            print('best model saved: step = ',global_step, 'dev RMSE = ',total_rmse, 'dev MAE = ', total_mae)

    print("\n[Validation Results]")
    print("Global Steps: %d" % global_step)
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid RMSE: %2.5f" % total_rmse)
    print("Valid MAE: %2.5f" % total_mae)
    print("time stamp: {}".format((time.time()-init_t)))

    return best_dev_accu

##################### FIXME: 스케줄러 ValueError -> 우선은 제외함. #####################
# def train(model, optimizer, lr_scheduler, ds_iter, training_config, criterion):
def train(model, optimizer, ds_iter, training_config, criterion):

    logger.info("***** Running training *****")
    logger.info("  Total steps = %d", training_config["num_train_steps"])
    losses = AverageMeter()

    checkpoint_path = training_config['checkpoint_path']
    best_dev_rmse = 0

    total_epochs = training_config["num_epochs"]
    epoch_iterator = tqdm(ds_iter['train'],
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
    
    model.train()
    init_t = time.time()
    total_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # Training step
    for epoch in range(total_epochs):
        for step, batch in enumerate(epoch_iterator):
            # 모델의 입력은 batch 그 자체, batch는 Dict이며 따라서 Dict 안의 tensor들을 device로 load.
            batch['user_seq'] = batch['user_seq'].cuda()
            batch['user_degree'] = batch['user_degree'].cuda()
            batch['item_list'] = batch['item_list'].cuda()
            batch['item_degree'] = batch['item_degree'].cuda()
            batch['item_rating'] = batch['item_rating'].cuda()
            batch['spd_matrix'] = batch['spd_matrix'].cuda()

            # forward pass
            outputs = model(batch)

            # [batch_size, seq_len_item, seq_len_user] : model output
            # [batch_size, seq_len_user, seq_len_item] : target인 rating matrix 

            # compute loss 
            loss = criterion(outputs.float(), batch['item_rating'].float())

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
            optimizer.step()

            ##################### FIXME: 스케줄러 ValueError -> 우선은 제외함. #####################
            # lr_scheduler.step()
            ##################### FIXME: 스케줄러 ValueError -> 우선은 제외함. #####################

            losses.update(loss)
            epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), losses.val))

        # Validation step
        if (step + 1) % training_config["eval_frequency"] == 0:
            end.record()
            torch.cuda.synchronize()
            total_time += (start.elapsed_time(end))
            best_dev_rmse = valid(model, ds_iter, training_config, checkpoint_path, step, best_dev_rmse, init_t, criterion)
            model.train()
            start.record()


    print("total training time (s): {}".format((time.time()-init_t)))
    print("total training time (ms): {}".format(total_time))
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("total memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    print(torch.cuda.memory_summary(device=0))


def eval(model, ds_iter, criterion):

    val_rmse = []
    val_mae = []
    eval_losses = AverageMeter()
    model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _, batch in ds_iter['test']:
            
            # 모델의 입력은 batch 그 자체, batch는 Dict이며 따라서 Dict 안의 tensor들을 device로 load.
            batch['user_seq'] = batch['user_seq'].cuda()
            batch['user_degree'] = batch['user_degree'].cuda()
            batch['item_list'] = batch['item_list'].cuda()
            batch['item_degree'] = batch['item_degree'].cuda()
            batch['item_rating'] = batch['item_rating'].cuda()
            batch['spd_matrix'] = batch['spd_matrix'].cuda()
            outputs = model(batch)

            loss = criterion(outputs.float(), batch['item_rating'].float())
            # eval_losses.update(loss.mean())
            eval_losses.update(loss)
            
            mse = F.mse_loss(outputs, batch['item_rating'], reduction='none')
            rmse = torch.sqrt(mse.mean())
            mae = F.l1_loss(outputs, batch['item_rating'], reduction='mean')

            val_rmse.append(rmse)
            val_mae.append(mae)

        total_rmse = sum(val_rmse) / len(val_rmse)
        total_mae = sum(val_mae) / len(val_mae)

    end.record()
    torch.cuda.synchronize()

    print("[Evaluation Results]")
    print("Loss: %2.5f" % eval_losses.avg)
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)
    print(f"total eval time: {(start.elapsed_time(end))}")
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))

    
def get_args():
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')
    parser.add_argument("--mode", type = str, default="train",
                        help="train eval")
    parser.add_argument("--dataset", type = str, default="ciao",
                        help = "ciao, epinions")
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, help="checkpoint model name")
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--n_layers', type=int, default=12)
    # parser.add_argument('--lr', type=float, default=2e-4)
    # parser.add_argument('--warmup', type=int, default=10000)
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--save_dir', type=str, default='./ckpts')
    # parser.add_argument('--hidden_size', type=int, default=128)
    # parser.add_argument('--ff_hidden_size', type=int, default=512)
    # parser.add_argument('--num_head', type=int, default=4)
    # parser.add_argument('--encoder_seq_length', type=int, default=20)
    # parser.add_argument('--decoder_seq_length', type=int, default=20)
    # parser.add_argument('--total_step', type=int, default=100000)
    # parser.add_argument('--optimizer', type=str, default='adam')
    # parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    ### get model config ###
    model_config = Config[args.dataset]["model"]
    training_config = Config[args.dataset]["training"]
    #training_config["learning_rate"] = args.learning_rate

    ### log preparation ###
    log_dir = os.getcwd() + f'/logs/log_seed_{args.seed}/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.name))
    redirect_stdout(open(log_path, 'w'))

    print(json.dumps([model_config, training_config], indent = 4))

    ###  set the random seeds for deterministic results. ####
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    ### model preparation ###
    model = Transformer(**model_config)

    checkpoint_dir = os.getcwd() + f'/checkpoints/checkpoints_seed_{args.seed}/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, args.mode)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'{args.name}.model')
    training_config["checkpoint_path"] = checkpoint_path
    """if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("model loaded from: " + checkpoint_path)"""


    #model = model.cuda()
    print(model)
    print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

    device_ids = list(range(torch.cuda.device_count()))
    print(f"GPU list: {device_ids}")
    model = nn.DataParallel(model, device_ids = device_ids).cuda()

    ### data preparation ###

    # ds_iter = {
    #         "train":DataLoader(MyDataset(f"./dataset/{args.dataset}.train.pickle", True), batch_size = training_config["batch_size"], shuffle=True),
    #         "dev":enumerate(DataLoader(MyDataset(f"./data/{args.dataset}.dev.pickle", True), batch_size = training_config["batch_size"]), shuffle=False),
    #         "test":enumerate(DataLoader(MyDataset(f"./data/{args.dataset}.test.pickle", False), batch_size = training_config["batch_size"]), shuffle=False),
    # }

    ### FIXME: 전체 데이터에 대해 파일 생성이 오래 걸림 (현재 시퀀스의 rating matrix 생성하는 부분이 문제로 보임) ==> rw sequence의 시작 노드 번호가 user_id=100 까지만 생성한 것으로 test.
    train_ds = MyDataset(dataset=args.dataset, split='train')
    dev_ds = MyDataset(dataset=args.dataset, split='valid')
    test_ds = MyDataset(dataset=args.dataset, split='test')

    ds_iter = {
            "train":DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=True),
            "dev":enumerate(DataLoader(dev_ds, batch_size = training_config["batch_size"], shuffle=True)),
            "test":enumerate(DataLoader(test_ds, batch_size = training_config["batch_size"], shuffle=False))
    }

    ### training preparation ###

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = training_config["learning_rate"],
        betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
    )

    # total_steps는 cycle당 있는 step 수. 없다면 epoch와 steps_per_epoch를 전댈해야함.
        # steps_per_epoch는 한 epoch에서의 전체 step 수: (total_number_of_train_samples / batch_size)
    total_epochs = training_config["num_epochs"]
    total_train_samples = len(train_ds)
    training_config["num_train_steps"] = math.ceil(total_train_samples / total_epochs)
    
    ###################### FIXME: num_epochs 1인 경우는 OK. 늘어나는 경우 ValueError: Tried to step 92 times. The specified number of total steps is 90 발생함 (10일경우)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer = optimizer,
    #     max_lr = training_config["learning_rate"],
    #     pct_start = training_config["warmup"] / training_config["num_train_steps"],
    #     anneal_strategy = training_config["lr_decay"],
    #     total_steps = training_config["num_train_steps"]
    # )
    ######################

    criterion = nn.MSELoss()

    ### train ###
    if args.mode == 'train':
        # train(model, optimizer, lr_scheduler, ds_iter, training_config, criterion)
        train(model, optimizer, ds_iter, training_config, criterion)

    ### eval ###
    if os.path.exists(checkpoint_path) and checkpoint_path != os.getcwd() + '/checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
    eval(model, ds_iter, criterion)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()