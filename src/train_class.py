import logging
import os

from pathlib import Path

from src.helpers import *


def train_classification(
    config_base: str, 
    config_classifier: str,
    n_iters: int=1,
    restart: bool=False,
):
    restart = restart or (n_iters!=1)

    root = str(Path(__file__).resolve().parents[1])
    results_dir = os.path.join(root, "results", config_base, config_classifier)
    models_dir = os.path.join(results_dir, "models")
    create_directory(models_dir)

    create_directory(os.path.join(root, "logs"))
    logs_path = "{}/logs/train_{}_{}.log".format(root, config_base, config_classifier)
    setup_default_logging(log_path=logs_path, restart=restart)
    logger = logging.getLogger('train')

    cfg_data, cfg_encoder, cfg_classifier, cfg_optimizer, cfg_scheduler = read_configs_model(
        path=os.path.join(root, "configs"),
        config_base=config_base,
        config_classifier=config_classifier,
        results_dir=results_dir,
        save=True,
    )
    device = get_device()

    train_datasets, test_datasets, n_classes = get_dataset(
        stats_dir=results_dir, **cfg_data
    )
    
    for fold, train_dataset in enumerate(train_datasets):

        logger.info("Fold {}".format(fold+1))

        test_loader, _ = get_loader(
            test_datasets[fold], batch_size=cfg_optimizer['batch_size']
        )

        cfg_encoder['input_fdim'], cfg_encoder['input_tdim'] = train_dataset.get_fbank_shape()
        if 'mixup' in cfg_data and cfg_data['mixup']>0:
            cfg_classifier['soft_labels'] = True 

        for iter in range(n_iters):

            logger.info("Iteration {}".format(iter+1))
            create_directory(os.path.join(models_dir, "fold_"+str(fold+1), "iter_"+str(iter+1)))

            train_loader, n_iters_train = get_loader(
                train_dataset, batch_size=cfg_optimizer['batch_size'], shuffle=True,
            )

            model = get_model(
                device=device,
                n_classes=n_classes,
                cfg_encoder=cfg_encoder,
                cfg_classifier=cfg_classifier,
            )
            logger.info("Number of parameters: {}".format(count_parameters(model)))

            optimizer, scheduler, n_epochs = get_optimizer_scheduler(
                model=model, 
                total_batches=n_iters_train,
                cfg_optimizer=cfg_optimizer,
                cfg_scheduler=cfg_scheduler,
            )

            ini_epoch = load_last_epoch_model(
                model_dir=os.path.join(models_dir, "fold_"+str(fold+1), "iter_"+str(iter+1)),
                model=model,
                optimizer=optimizer,
                restart=restart,
            )

            for epoch in range(ini_epoch+1, n_epochs+1):

                logger.info("\nEpoch {}/{}".format(epoch, n_epochs))

                train_epoch(
                    epoch=epoch, 
                    device=device,
                    train_loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_dir=os.path.join(
                        models_dir, "fold_"+str(fold+1), "iter_"+str(iter+1)
                    ) if epoch%10==0 else None
                )

                test_epoch(
                    logger=logger,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    model=model,
                    preds_dir=os.path.join(
                        results_dir, "preds", "fold_"+str(fold+1), "iter_"+str(iter+1), "epoch_"+str(epoch)
                    ) if epoch%10==0 else None
                )