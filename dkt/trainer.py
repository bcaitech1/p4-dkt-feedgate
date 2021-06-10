import os
from numpy.lib.function_base import diff
import torch
import numpy as np

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import *

import wandb

def run(args, train_data, valid_data):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
            
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
        
        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                  "valid_auc":auc, "valid_acc":acc})
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.model_dir, f'{args.name}.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        preds = model(input)
        # targets = input[4] # correct
        targets = input[8] # correct


        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)
      

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        # targets = input[4] # correct
        targets = input[8] # correct


        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]
    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets



def inference(args, test_data):
    
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)

        preds = model(input)
        

        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    write_path = os.path.join(args.output_dir, f"{args.name}.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))




def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    model_class = str_to_class(args.model)
    model = model_class(args)
    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):

    # test, category, number, tag, correct, mask = batch
    # test, category, number, tag, time, correct, mask = batch
    test, category, number, tag, soltime, time, average_prob_correct_cate, past_user_prob_count, correct, mask = batch
    # test, category, number, tag, clipped_soltime, time, correct, mask = batch
    
    
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction[:, 0] = 0 # set padding index to the first sequence
    interaction = (interaction * mask).to(torch.int64)
    # print(interaction)
    # exit()

    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    # question = ((question + 1) * mask).to(torch.int64)
    category = ((category + 1) * mask).to(torch.int64)
    number = ((number + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)
    soltime = ((soltime + 1) * mask).to(torch.int64)
    # clipped_soltime = ((clipped_soltime + 1) * mask).to(torch.int64)
    time = ((time + 1) * mask).to(torch.int64)
    # sol_num = ((sol_num + 1) * mask).to(torch.int64)

    # hour = ((hour + 1) * mask).to(torch.int64)
    # weekday = ((weekday+ 1) * mask).to(torch.int64)
    # average_user_correct = ((average_user_correct + 1) * mask).to(torch.int64)
    # average_tag_correct = ((average_tag_correct + 1) * mask).to(torch.int64)
    # average_prob_correct = ((average_prob_correct + 1) * mask).to(torch.int64)
    average_prob_correct_cate = ((average_prob_correct_cate + 1) * mask).to(torch.int64)
    # average_user_correct_cate = ((average_user_correct_cate + 1) * mask).to(torch.int64)
    # past_prob_count = ((past_prob_count + 1) * mask).to(torch.int64)
    past_user_prob_count = ((past_user_prob_count + 1) * mask).to(torch.int64)
    

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    # device memory로 이동

    test = test.to(args.device)
    # question = question.to(args.device)
    category = category.to(args.device)
    number = number.to(args.device)

    tag = tag.to(args.device)
    soltime = soltime.to(args.device)
    # clipped_soltime = clipped_soltime.to(args.device)
    time = time.to(args.device)
    # sol_num = sol_num.to(args.device)

    correct = correct.to(args.device)
    # hour = hour.to(args.device)
    # weekday = weekday.to(args.device)
    # average_user_correct = average_user_correct.to(args.device)
    # average_tag_correct = average_tag_correct.to(args.device)
    # average_prob_correct = average_prob_correct.to(args.device)
    average_prob_correct_cate = average_prob_correct_cate.to(args.device)
    # average_user_correct_cate = average_user_correct_cate.to(args.device)
    # past_prob_count = past_prob_count.to(args.device)
    past_user_prob_count = past_user_prob_count.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    # return (test, category, number,
    #         tag,  correct, mask,
    #         interaction, gather_index)
    # return (test, category, number,
    #         tag, time,
    #         correct, mask,
    #         interaction, gather_index)
    return (test, category, number,
            tag, soltime, time, average_prob_correct_cate, past_user_prob_count,
            correct, mask,
            interaction, gather_index)
    # return (test, category, number,
    #         tag, clipped_soltime, time,
    #         correct, mask,
    #         interaction, gather_index)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)
    """
    loss = get_criterion(preds, targets)
    #마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:,-1]
    loss = torch.mean(loss)
    return loss

def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()



def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))



def load_model(args):
    
    
    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
   
    
    print("Loading Model from:", model_path, "...Finished.")
    return model