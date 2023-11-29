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
from torch.utils.tensorboard import SummaryWriter

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

# def MaskedMSELoss(target, prediction):
#     """
#     Compute Masked MSELoss
#     """
#     mask = (target != 0).float()
#     squared_diff = (prediction - target)**2 * mask
#     loss = torch.sum(squared_diff) / torch.sum(mask)

#     return loss

def valid(model, ds_iter, epoch, checkpoint_path, global_step, best_dev_rmse, best_dev_mae, init_t):
    val_rmse = []
    val_mae = []
    eval_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        # FIXME: valid를 기준으로 저장 X, test를 기준으로 바로 저장. 
        epoch_iterator = tqdm(ds_iter['test'],
                              desc="Validating (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              leave=False)
        for step, batch in enumerate(epoch_iterator):
            batch['user_seq'] = batch['user_seq'].cuda()
            batch['user_degree'] = batch['user_degree'].cuda()
            batch['item_list'] = batch['item_list'].cuda()
            batch['item_degree'] = batch['item_degree'].cuda()
            batch['item_rating'] = batch['item_rating'].cuda()
            batch['spd_matrix'] = batch['spd_matrix'].cuda()

            outputs = model(batch)

            # loss = criterion(outputs.float(), batch['item_rating'].float())
            # FIXME:
                # 현재 target(batch['item_rating])은 0이 많이 포함되어 있는 sparse한 rating matrix로, shape가 [seq_len_user, seq_len_item] (batch dim 제외)
                # model output 또한 마찬가지로 shape가 [seq_len_user, seq_len_item].
                # 단순히 이 둘의 MSELoss를 계산하는 경우, known rating에 대한 제곱오차만을 계산하는 것이 아닌 unknown rating(0)에 대한 제곱오차도 계산하게 됨.
                # 따라서 Masked MSELoss를 사용.
                # model의 출력에서 unknown rating에 대한 부분을 0으로 masking 처리, 제곱오차 계산 시 known rating과 만의 제곱오차를 계산.
            mask = (batch['item_rating'] != 0)
            squared_diff = (outputs - batch['item_rating'])**2 * mask
            loss = torch.sum(squared_diff) / torch.sum(mask)

            eval_losses.update(loss)

            # 실제 rating matrix에서 0이 아닌 부분(실제 매긴 rating)과만 loss를 계산
                # -> 현재 목표는 rating regression이기 때문이니까.
            mse = F.mse_loss(outputs[mask], batch['item_rating'][mask], reduction='none')
            rmse = torch.sqrt(mse.mean())
            mae = F.l1_loss(outputs[mask], batch['item_rating'][mask], reduction='mean')

            val_rmse.append(rmse)
            val_mae.append(mae)

            epoch_iterator.set_description(
                        "Validating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), eval_losses.val))
        
        total_rmse = sum(val_rmse) / len(val_rmse)
        total_mae = sum(val_mae) / len(val_mae)

        if total_rmse < best_dev_rmse:
            best_dev_rmse = total_rmse
            best_dev_mae = total_mae
            torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
            print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse.item():.6f}, test MAE = {total_mae.item():.6f}')

    return eval_losses.avg, best_dev_rmse, best_dev_mae, total_rmse, total_mae

def train(model, optimizer, lr_scheduler, ds_iter, training_config, writer):
# def train(model, optimizer, ds_iter, training_config, criterion):

    # TODO: Epoch당 loss, RMSE, MAE 추적 => TensorBoard 또는 파일 저장을 통해 tracing할 수 있도록.
    logger.info("***** Running training *****")
    logger.info("  Total steps = %d", training_config["num_train_steps"])
    losses = AverageMeter()

    checkpoint_path = training_config['checkpoint_path']
    best_dev_rmse = 9999.0
    best_dev_mae = 9999.0

    total_epochs = training_config["num_epochs"]

    model.train()
    init_t = time.time()
    total_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # Training step
    for epoch in range(total_epochs):
        epoch_iterator = tqdm(ds_iter['train'],
                            desc="Training (X / X Steps) (loss=X.X)",
                            bar_format="{l_bar}{r_bar}",
                            dynamic_ncols=True,
                            leave=False)
        
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

            # compute loss
            # FIXME: 
                # 현재 target(batch['item_rating'])은 0이 많이 포함되어 있는 sparse한 rating matrix로, shape가 [seq_len_user, seq_len_item] (batch dim 제외)
                # model output 또한 마찬가지로 shape가 [seq_len_user, seq_len_item].
                # 단순히 이 둘의 MSELoss를 계산하는 경우, known rating에 대한 제곱오차만을 계산하는 것이 아닌 unknown rating(0)에 대한 제곱오차도 계산하게 됨.
                # 따라서 Masked MSELoss를 사용
                # model의 출력에서 unknown rating에 대한 부분을 0으로 masking 처리, 제곱오차 계산 시 known rating과 만의 제곱오차를 계산하게 된다.
            # loss = criterion(outputs.float(), batch['item_rating'].float())
            mask = (batch['item_rating'] != 0)

            squared_diff = (outputs - batch['item_rating'])**2 * mask

            loss = torch.sum(squared_diff) / torch.sum(mask)

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
            optimizer.step()

            lr_scheduler.step()

            losses.update(loss)
            epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), losses.val))

        # validation
        end.record()
        torch.cuda.synchronize()
        total_time += (start.elapsed_time(end))
        valid_loss, best_dev_rmse, best_dev_mae, valid_rmse, valid_mae = valid(model, ds_iter, epoch, checkpoint_path, step, best_dev_rmse, best_dev_mae, init_t)
        model.train()
        start.record()

        # Tensorboard recording
        # writer.add_scalar('Loss/Train', losses.avg, epoch)
        # writer.add_scalar('Loss/Valid', valid_loss, epoch)
        writer.add_scalars('Loss', {'Train':losses.avg, 'Valid':valid_loss}, epoch)
        writer.add_scalar('RMSE/Test', valid_rmse, epoch)
        writer.add_scalar('MAE/Test', valid_mae, epoch)

        print(f"Epoch {epoch:03d}: Train Loss: {losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch RMSE: {valid_rmse:.4f} || epoch MAE: {valid_mae:.4f} || best RMSE: {best_dev_rmse:.4f} || best MAE: {best_dev_mae:.4f}")

    writer.close()

    print('\n [Train Finished]')
    print("total training time (s): {}".format((time.time()-init_t)))
    print("total training time (ms): {}".format(total_time))
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("total memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    print(torch.cuda.memory_summary(device=0))


def eval(model, ds_iter):

    val_rmse = []
    val_mae = []
    eval_losses = AverageMeter()
    model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        epoch_iterator = tqdm(ds_iter['test'],
                        desc="Validating (X / X Steps) (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True,
                        leave=False)
        # for _, batch in ds_iter['test']:
        for step, batch in enumerate(epoch_iterator):
            
            # 모델의 입력은 batch 그 자체, batch는 Dict이며 따라서 Dict 안의 tensor들을 device로 load.
            batch['user_seq'] = batch['user_seq'].cuda()
            batch['user_degree'] = batch['user_degree'].cuda()
            batch['item_list'] = batch['item_list'].cuda()
            batch['item_degree'] = batch['item_degree'].cuda()
            batch['item_rating'] = batch['item_rating'].cuda()
            batch['spd_matrix'] = batch['spd_matrix'].cuda()
            outputs = model(batch)

            # loss = criterion(outputs.float(), batch['item_rating'].float())
            # FIXME: 
                # 현재 target(batch['item_rating'])은 0이 많이 포함되어 있는 sparse한 rating matrix로, shape가 [seq_len_user, seq_len_item] (batch dim 제외)
                # model output 또한 마찬가지로 shape가 [seq_len_user, seq_len_item].
                # 단순히 이 둘의 MSELoss를 계산하는 경우, known rating에 대한 제곱오차만을 계산하는 것이 아닌 unknown rating(0)에 대한 제곱오차도 계산하게 됨.
                # 따라서 Masked MSELoss를 사용
                # model의 출력에서 unknown rating에 대한 부분을 0으로 masking 처리, 제곱오차 계산 시 known rating과 만의 제곱오차를 계산하게 된다.
            mask = (batch['item_rating'] != 0)
            squared_diff = (outputs - batch['item_rating'])**2 * mask
            loss = torch.sum(squared_diff) / torch.sum(mask)

            # eval_losses.update(loss.mean())
            eval_losses.update(loss)
            
            # 실제 rating matrix에서 0이 아닌 부분(실제 매긴 rating)과만 loss를 계산
                # -> 현재 목표는 rating regression이기 때문이니까.
            mse = F.mse_loss(outputs[mask], batch['item_rating'][mask], reduction='none')
            rmse = torch.sqrt(mse.mean())
            mae = F.l1_loss(outputs[mask], batch['item_rating'][mask], reduction='mean')

            val_rmse.append(rmse)
            val_mae.append(mae)

            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), eval_losses.val))

        total_rmse = sum(val_rmse) / len(val_rmse)
        total_mae = sum(val_mae) / len(val_mae)

    end.record()
    torch.cuda.synchronize()

    print("\n [Evaluation Results]")
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
    parser.add_argument('--user_seq_len', type=int, default=20, help="user random walk sequence length")
    parser.add_argument('--item_seq_len', type=int, default=250, help="item list length")
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

    # log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.name))
    # redirect_stdout(open(log_path, 'w'))

    # print(json.dumps(args.__dict__, indent = 4))

    # print(json.dumps([model_config, training_config], indent = 4))

    ###  set the random seeds for deterministic results. ####
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    ### model preparation ###
    model = Transformer(**model_config)

    # checkpoint_dir = os.getcwd() + f'/checkpoints/{args.dataset}/checkpoints_seed_{args.seed}/'
    checkpoint_data = os.getcwd() + f'/checkpoints/{args.dataset}/'
    if not os.path.exists(checkpoint_data):
        os.mkdir(checkpoint_data)
    checkpoint_dir = checkpoint_data + f'checkpoints_seed_{args.seed}/'
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
    # print(model)
    # print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
    # print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

    device_ids = list(range(torch.cuda.device_count()))
    print(f"GPU list: {device_ids}")
    model = model.cuda()

    ### data preparation ###

    ### FIXME: 전체 데이터에 대해 파일 생성이 오래 걸림 (현재 시퀀스의 rating matrix 생성하는 부분이 문제로 보임)
        ### FIXME: (231012) validation set을 통해 모델이 잘 train 되는것은 확인했으므로, 바로 test를 진행하면서 model을 저장.
    train_ds = MyDataset(dataset=args.dataset, split='train', seed=args.seed, user_seq_len=args.user_seq_len, item_seq_len=args.item_seq_len)
    # dev_ds = MyDataset(dataset=args.dataset, split='valid', seed=args.seed, user_seq_len=args.user_seq_len, item_seq_len=args.item_seq_len)
    test_ds = MyDataset(dataset=args.dataset, split='test', seed=args.seed, user_seq_len=args.user_seq_len, item_seq_len=args.item_seq_len)

    ds_iter = {
            "train":DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=4),
            # "dev":DataLoader(dev_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=4),
            "test":DataLoader(test_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=4)
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
    
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer = optimizer,
    #     max_lr = training_config["learning_rate"],
    #     pct_start = training_config["warmup"] / training_config["num_train_steps"],
    #     anneal_strategy = training_config["lr_decay"],
    #     total_steps = training_config["num_train_steps"]
    # )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer = optimizer,
        max_lr = training_config["learning_rate"],
        pct_start = training_config["warmup"] / training_config["num_train_steps"],
        anneal_strategy = training_config["lr_decay"],
        epochs = training_config["num_epochs"],
        steps_per_epoch = 2 * len(ds_iter['train'])
    )

    # criterion = nn.MSELoss()

    ### TensorBoard writer preparation ###
    writer = SummaryWriter(os.path.join(log_dir,f"{args.name}.tensorboard"))
    ### train ###
    if args.mode == 'train':
        train(model, optimizer, lr_scheduler, ds_iter, training_config, writer)
        # train(model, optimizer, ds_iter, training_config, criterion)

    # Since train logging is done by TensorBoard, log only test result.
    log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.name))
    redirect_stdout(open(log_path, 'w'))

    print(json.dumps(args.__dict__, indent = 4))

    print(json.dumps([model_config, training_config], indent = 4))

    print(model)
    print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

    ### eval ###
    if os.path.exists(checkpoint_path) and checkpoint_path != os.getcwd() + '/checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
    eval(model, ds_iter)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()