from dataset import AFADDataset
from model import GenderAge

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from apex import amp

import torch.nn as nn
import numpy as np
import shutil
import torch
import yaml
import time
import sys
import os


best_acc = float('-inf')
global_step = 0
num_steps = 0


def load_data(args):
    dataset = AFADDataset(args['DATASET'], args['ANNOTATION'], args['INPUT_SIZE'], True)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    val_size = int(args['VAL_RATIO'] * len(dataset))
    val_idx, train_idx = indices[: val_size], indices[val_size:]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=args['BS'], sampler=train_sampler, num_workers=args['NW'], pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=args['BS'], sampler=val_sampler, num_workers=args['NW'], pin_memory=True)
    data_loaders = {'train': train_loader, 'val': val_loader}
    return data_loaders


def show_lr(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr += param_group['lr']
    return lr


def train(model, data_loader, criterion, epoch, optimizer, apex):
    global global_step, num_steps
    num_steps = 0

    print('\n' + '-' * 10)
    print('Epoch: {}'.format(epoch))
    print('Current Learning rate: {}'.format(show_lr(optimizer)))

    model.train()

    timer = time.time()
    dataset_size = len(data_loader.dataset)
    train_gender_loss, train_gender_acc, train_age_loss, train_age_error, processed_size = 0, 0, 0, 0, 0

    for images, genders, ages in data_loader:
        # Forward
        torch.cuda.empty_cache()
        images, genders, ages = images.cuda(), genders.cuda(), ages.cuda()
        logits = model(images)
        # Compute loss
        gender_loss = criterion(logits[:, :2], genders)
        weights = torch.zeros(58).cuda()
        for i in range(15, 73):
            weights[i - 15] = torch.sum((torch.ones_like(ages) * i - ages) ** 2) / len(ages)
        age_loss = torch.sum((logits[:, 2] - ages) ** 2 / weights[torch.clip(logits[:, 2].long() - 15, 0, 57)]) / len(logits)
        # age_loss = mse_loss(logits[:, 2], ages)
        loss = gender_loss + age_loss * 100

        # Backward
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Record and display data
        processing_size = len(images)
        processed_size += processing_size

        batch_gender_loss = gender_loss.item()
        train_gender_loss += batch_gender_loss * processing_size

        batch_age_loss = age_loss
        train_age_loss += batch_age_loss * processing_size

        _, gender_preds = torch.max(logits.data[:, :2], 1)
        batch_gender_acc = (gender_preds == genders.data).sum().item() / processing_size
        train_gender_acc += batch_gender_acc * processing_size

        age_preds = logits.data[:, 2]
        batch_age_error = torch.abs(age_preds - ages.data).sum().item() / processing_size
        train_age_error += batch_age_error * processing_size

        train_writer.add_scalar('gender/loss', batch_gender_loss, global_step)
        train_writer.add_scalar('gender/acc', batch_gender_acc, global_step)
        train_writer.add_scalar('age/loss', batch_age_loss, global_step)
        train_writer.add_scalar('age/error', batch_age_error, global_step)
        sys.stdout.write('\rProcess: [{:5.0f}/{:5.0f} ({:2.2%})] '
                         'Gender loss: {:.4f}/{:.4f} '
                         'Gender acc: {:.2%}/{:.2%} '
                         'Age loss: {:.4f}/{:.4f} '
                         'Age error: {:.2f}/{:.2f} '
                         'Estimated time: {:.2f}s'.format(
                            processed_size, dataset_size, processed_size / dataset_size,
                            float(batch_gender_loss), float(train_gender_loss) / processed_size,
                            float(batch_gender_acc), float(train_gender_acc) / processed_size,
                            float(batch_age_loss), float(train_age_loss) / processed_size,
                            float(batch_age_error), train_age_error / processed_size,
                            (time.time() - timer))),
        sys.stdout.flush()
        global_step += 1
        num_steps += 1
        timer = time.time()

    # Record and display data
    print('\nTrain Gender Loss: {:.4f} Train Gender Acc: {:.2%} Train Age Loss: {:.4f} Train Age Error: {:.2f}'.format(
        train_gender_loss / processed_size, train_gender_acc / processed_size, train_age_loss / processed_size, train_age_error / processed_size))


def val(model, data_loader, criterion, epoch, save):
    global best_acc

    model.eval()

    with torch.no_grad():
        val_gender_loss, val_gender_acc, val_age_loss, val_age_error, processed_size = 0, 0, 0, 0, 0
    
        for images, genders, ages in data_loader:
            # Forward
            torch.cuda.empty_cache()
            images, genders, ages = images.cuda(), genders.cuda(), ages.cuda()
            logits = model(images)
    
            # Record and display data
            processing_size = len(images)
            processed_size += processing_size
            
            gender_loss = criterion(logits[:, :2], genders)
            val_gender_loss += gender_loss.item() * processing_size
            _, gender_preds = torch.max(logits.data[:, :2], 1)
            val_gender_acc += (gender_preds == genders.data).sum().item()

            a = torch.zeros(58).cuda()
            for i in range(15, 73):
                a[i - 15] = torch.sum((torch.ones_like(ages) * i - ages) ** 2) / len(ages)
            age_loss = torch.sum((logits[:, 2] - ages) ** 2 / a[logits[:, 2].long() - 15]) / len(logits)
            # age_loss = mse_loss(logits[:, 2], ages)
            val_age_loss += age_loss * processing_size
            age_preds = logits.data[:, 2]
            val_age_error += torch.abs(age_preds - ages.data).sum().item()

        # Record and display data
        val_gender_loss /= processed_size
        val_gender_acc /= processed_size
        val_age_loss /= processed_size
        val_age_error /= processed_size
        val_writer.add_scalar('gender/loss', val_gender_loss, epoch * num_steps)
        val_writer.add_scalar('gender/acc', val_gender_acc, epoch * num_steps)
        val_writer.add_scalar('age/loss', val_age_loss, epoch * num_steps)
        val_writer.add_scalar('age/error', val_age_error, epoch * num_steps)
        print('Val Gender Loss: {:.4f} Val Gender Acc: {:.2%} Val Age Loss: {:.4f} Val Age Error: {:.2f}'.format(
            val_gender_loss, val_gender_acc, val_age_loss, val_age_error))

        # Save model
        if save and val_gender_acc > best_acc:
            best_acc = max(val_gender_acc, best_acc)
            shutil.rmtree(save_path)
            os.makedirs(save_path)
            torch.save(model, os.path.join(save_path, 'best-epoch{}-{:.4f}.pt'.format(epoch, best_acc)))


def main(args):
    model = GenderAge(args['NC'], args['LAYERS'])
    if args['MODEL']:
        model = torch.load(args['MODEL'])
    else:
        print('train from scratch')
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['LR'], weight_decay=5e-5)
    if args['APEX']:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['STEP_SIZE'], gamma=args['GAMMA'])
    criterion = nn.CrossEntropyLoss()

    data_loaders = load_data(args)
    
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:{:,} .'.format(total_params))
    print('total_trainable_parameters:{:,} .'.format(total_trainable_params))
    
    val(model, data_loaders['val'], criterion, 0, False)
    for epoch in range(args['NE']):
        torch.cuda.empty_cache()
        train(model, data_loaders['train'], criterion, epoch + 1, optimizer, args['APEX'])
        val(model, data_loaders['val'], criterion, epoch + 1, True)
        scheduler.step()


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    train_writer = SummaryWriter(log_dir=os.path.join('logs-' + time.strftime("%Y-%m-%d-%H-%M", time.localtime()), 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join('logs-' + time.strftime("%Y-%m-%d-%H-%M", time.localtime()), 'val'))

    save_path = './models-' + time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    main(config)
