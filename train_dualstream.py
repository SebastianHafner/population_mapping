import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_dual_training(dual_cfg: experiment_manager.CfgNode):
    cfg1, cfg2 = dual_cfg.CFG1, dual_cfg.CFG2
    run_config = {
        'CONFIG_NAME': dual_cfg.NAME,
        'device': device,
        'epochs': dual_cfg.CFG1.TRAINER.EPOCHS,
        'learning rate': dual_cfg.CFG1.TRAINER.LR,
        'batch size': dual_cfg.CFG1.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    dual_net = networks.DualStreamPopulationNet(cfg1, cfg2)
    dual_net.to(device)
    optimizer = optim.AdamW(dual_net.parameters(), lr=dual_cfg.CFG1.TRAINER.LR, weight_decay=0.01)
    criterion = loss_functions.get_criterion(cfg1.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = datasets.CellDualInputPopulationDataset(dual_cfg=dual_cfg, run_type='train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg1.TRAINER.BATCH_SIZE,
        'num_workers': 0 if dual_cfg.DEBUG else cfg1.DATALOADER.NUM_WORKER,
        'shuffle': dual_cfg.CFG1.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg1.TRAINER.EPOCHS
    save_checkpoints = cfg1.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set, pop_set = [], []

        for i, (batch) in enumerate(dataloader):

            dual_net.train()
            dual_net.zero_grad()

            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            gt = batch['y'].to(device)
            pred, _, _ = dual_net(x1, x2)

            loss = criterion(pred, gt.float())
            loss.backward()
            optimizer.step()

            loss.append(loss.item())
            loss_set.append(loss.item())
            pop_set.append(gt.flatten())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg1.LOG_FREQ == 0 and not dual_cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'train', epoch_float, global_step,
                                                            max_samples=1_000)
                evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'test', epoch_float, global_step,
                                                            max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                pop_set = torch.cat(pop_set)
                mean_pop = torch.mean(pop_set)
                null_percentage = torch.sum(pop_set == 0) / torch.numel(pop_set) * 100
                wandb.log({
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 100,
                    'mean_population': mean_pop,
                    'null_percentage': null_percentage,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set, pop_set = [], []

            if dual_cfg.DEBUG:
                # testing evaluation
                evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'train', epoch_float, global_step,
                                                            max_samples=1_000)
                evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'test', epoch_float, global_step,
                                                            max_samples=1_000)
                break
            # end of batch

        if not dual_cfg.DEBUG:
            assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')

        if epoch in save_checkpoints and not dual_cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(dual_net, optimizer, epoch, global_step, cfg1)

            # logs to load network
            evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'train', epoch_float, global_step)
            evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'test', epoch_float, global_step)
            for city in dual_cfg.CFG1.DATASET.CENSUS_EVALUATION_CITIES:
                print(f'Running census-level evaluation for {city}...')
                evaluation.model_evaluation_census_dualstream(dual_net, dual_cfg, city)


if __name__ == '__main__':

    args = parsers.dualstream_argument_parser().parse_known_args()[0]
    cfg1 = experiment_manager.setup_cfg(args, config_name=args.config_file1)
    cfg2 = experiment_manager.setup_cfg(args, config_name=args.config_file2)

    # make training deterministic
    torch.manual_seed(cfg1.SEED)
    np.random.seed(cfg1.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    dual_cfg = experiment_manager.CfgNode()
    dual_cfg.CFG1 = cfg1
    dual_cfg.CFG2 = cfg2
    dual_cfg.NAME = f'dualstream_{cfg1.NAME}_{cfg2.NAME}'
    dual_cfg.STREAM_LOSSES_ENABLED = False
    dual_cfg.DEBUG = True if args.debug == 'True' else False
    wandb.init(
        name=dual_cfg.NAME,
        config=dual_cfg,
        entity='population_mapping',
        tags=['run', 'population', 'mapping', 'regression', ],
        mode='online' if not dual_cfg.DEBUG else 'disabled',
    )

    try:
        run_dual_training(dual_cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
