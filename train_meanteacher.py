import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager


def run_mean_teacher_training(net, cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    supervised_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    student_net = net
    teacher_net = networks.create_ema_network(student_net, cfg)
    student_net.to(device)
    teacher_net.to(device)
    consistency_criterion = loss_functions.get_criterion(cfg.CONSISTENCY_TRAINER.CONSISTENCY_LOSS_TYPE)

    dataset = datasets.PopulationMappingDataset(cfg=cfg, run_type='training', include_unlabeled=True)
    print(dataset)
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}.')

        start = timeit.default_timer()
        loss_set, supervised_loss_set, consistency_loss_set, pop_set = [], [], [], []
        n_labeled, n_notlabeled = 0, 0

        for i, batch in enumerate(dataloader):

            student_net.train()
            teacher_net.train()

            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            pop_set.append(y_gts.flatten())
            is_labeled = batch['is_labeled']

            y_pred_student = student_net(x)
            y_pred_teacher = teacher_net(x)
            y_pred_teacher = y_pred_teacher.detach()

            supervised_loss, consistency_loss = None, None

            if is_labeled.any():
                supervised_loss = supervised_criterion(y_pred_student[is_labeled,], y_gts[is_labeled])
                supervised_loss_set.append(supervised_loss.item())
                n_labeled += torch.sum(is_labeled).item()

            if not is_labeled.all():
                not_labeled = torch.logical_not(is_labeled)
                consistency_loss = consistency_criterion(y_pred_student[not_labeled,], y_pred_teacher[not_labeled,])
                consistency_loss_set.append(consistency_loss.item())
                n_notlabeled += torch.sum(not_labeled).item()

            if supervised_loss is None and consistency_loss is not None:
                loss = cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
            elif supervised_loss is not None and consistency_loss is not None:
                loss = supervised_loss + cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
            else:
                loss = supervised_loss

            loss_set.append(loss.item())
            loss.backward()
            optimizer.step()
            teacher_net.update()
            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation(net, cfg, 'training', epoch_float, global_step, max_samples=1_000)
                evaluation.model_evaluation(net, cfg, 'test', epoch_float, global_step, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                labeled_percentage = n_labeled / (n_labeled + n_notlabeled) * 100
                pop_set = torch.cat(pop_set)
                mean_pop = torch.mean(pop_set)
                null_percentage = torch.sum(pop_set == 0) / torch.numel(pop_set) * 100
                wandb.log({
                    'loss': np.mean(loss_set),
                    'supervised_loss': 0 if len(supervised_loss_set) == 0 else np.mean(supervised_loss_set),
                    'consistency_loss': 0 if len(consistency_loss_set) == 0 else np.mean(consistency_loss_set),
                    'labeled_percentage': labeled_percentage,
                    'mean_population': mean_pop,
                    'null_percentage': null_percentage,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })

                # resetting stuff
                start = timeit.default_timer()
                loss_set, supervised_loss_set, consistency_loss_set, pop_set = [], [], [], []
                n_labeled, n_notlabeled = 0, 0

            if cfg.DEBUG:
                # testing evaluation
                evaluation.model_evaluation(net, cfg, 'training', epoch_float, global_step, max_samples=1_000)
                evaluation.model_evaluation(net, cfg, 'test', epoch_float, global_step, max_samples=1_000)
                break
            # end of batch

        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)

            # logs to load network
            evaluation.model_evaluation(net, cfg, 'training', epoch_float, global_step)
            evaluation.model_evaluation(net, cfg, 'test', epoch_float, global_step)


if __name__ == '__main__':

    args = experiment_manager.default_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            config=cfg,
            entity='population_mapping',
            tags=['run', 'population', 'mapping', 'regression', ],
        )

    try:
        run_mean_teacher_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
