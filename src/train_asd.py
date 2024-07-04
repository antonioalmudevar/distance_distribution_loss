import logging
import os

from pathlib import Path

from src.helpers import *
from .datasets import get_dataset, get_datasets_ood


POSTPROCESSORS = ["KNN", "Mahalanobis"]


def train_model(
    config_data: str, 
    config_encoder: str, 
    config_classifier: str,
    config_training: str,
    n_iters: int=1,
    restart: bool=False,
    save_all_results: bool=False,
):
    restart = restart or (n_iters!=1)

    root = str(Path(__file__).resolve().parents[1])
    subroot = os.path.join(config_data, config_encoder, config_classifier, config_training)
    models_dir = os.path.join(root, "results", "models", subroot)
    create_directory(models_dir)
    preds_dir = os.path.join(root, "results", "preds", subroot)

    create_directory(os.path.join(root, "logs"))
    logs_path = "{}/logs/train_{}_{}_{}_{}.log".format(
        root, config_data, config_encoder, config_classifier, config_training
    )
    setup_default_logging(log_path=logs_path, restart=restart)
    logger = logging.getLogger('train')

    cfg_data, cfg_encoder, cfg_classifier, cfg_training = read_configs_model(
        path=os.path.join(root, "configs"),
        config_data=config_data,
        config_encoder=config_encoder,
        config_classifier=config_classifier,
        config_training=config_training,
        model_dir=models_dir,
        save=True,
    )

    device = get_device()

    train_dataset, n_classes = get_dataset(
        mode="train", 
        classical_augment=not is_prototypical(cfg_classifier['model_type']),
        **cfg_data
    )
    eval_dataset, _ = get_dataset(mode="eval", **cfg_data)
    val_dataset, _ = get_dataset(mode="val", **cfg_data)
    ood_datasets = get_datasets_ood(**cfg_data)

    eval_loader = get_loader(
        eval_dataset, batch_size=cfg_training['batch_size']
    )[0]
    val_loader = get_loader(
        val_dataset, batch_size=cfg_training['batch_size']
    )[0]
    ood_loaders = {
        dataset: get_loader(
            ood_datasets[dataset], batch_size=cfg_training['batch_size']
        )[0] for dataset in ood_datasets
    }

    for iter in range(n_iters):

        logger.info("Iteration {}".format(iter+1))
        create_directory(os.path.join(models_dir, "iter_"+str(iter)))

        train_loader, n_iters_train = get_loader(
            train_dataset, batch_size=cfg_training['batch_size'], shuffle=True,
        )

        model = get_model(
            device=device,
            n_classes=n_classes,
            loader=eval_loader,
            cfg_encoder=cfg_encoder,
            cfg_classifier=cfg_classifier,
        )
        logger.info("Number of parameters: {}".format(count_parameters(model)))

        optimizer, scheduler, n_epochs = get_optimizer_scheduler(
            model=model, 
            total_batches=n_iters_train,
            cfg_optimizer=cfg_training['optimizer'],
            cfg_scheduler=cfg_training['scheduler'],
        )

        ini_epoch = load_last_epoch_model(
            model_dir=os.path.join(models_dir, ("iter_"+str(iter))),
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
                    models_dir, "iter_"+str(iter)
                ) if epoch%10==0 else None
            )

            test_epoch(
                logger=logger,
                device=device,
                train_loader=eval_loader,
                test_loader=val_loader,
                ood_loaders=ood_loaders,
                model=model,
                is_prototypical=is_prototypical(cfg_classifier['model_type']),
                calc_ood_metrics=(epoch%n_epochs==0),
                preds_dir=os.path.join(
                    preds_dir, "iter_"+str(iter), "epoch_"+str(epoch)
                ) if epoch%10==0 else None
            )