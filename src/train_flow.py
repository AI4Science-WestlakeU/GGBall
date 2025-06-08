import graph_tool as gt
import os
from pathlib import Path
import warnings
from typing import Any, List
import torch
import pytorch_lightning as pl
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf
from models.flow_utils import register_omega_conf_resolvers
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.HGCN import HGCN
from src.HGVAE import HGVAE
from src.HypeFlow import ManifoldFMLitModule
import pdb
import numpy as np

warnings.filterwarnings("ignore", category=PossibleUserWarning)

try:
    WANDB_MODE = os.environ["WANDB_MODE"]
except KeyError:
    WANDB_MODE = ""


register_omega_conf_resolvers()


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if (WANDB_MODE.lower() != "disabled") and ("lr_monitor" in cfg.logging):
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k * 2,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

        callbacks.append(
            ModelCheckpoint(
                monitor='val/log_metric',
                mode='min',
                save_top_k=cfg.train.model_checkpoints.save_top_k * 2,
                verbose=True,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

        callbacks.append(
            ModelCheckpoint(
                monitor='val/log_metric_mean',
                mode='min',
                save_top_k=5,
                verbose=True,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    if "every_n_epochs_checkpoint" in cfg.train:
        hydra.utils.log.info(
            f"Adding callback <ModelCheckpoint> for every {cfg.train.every_n_epochs_checkpoint.every_n_epochs} epochs"
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath="every_n_epochs",
                every_n_epochs=cfg.train.every_n_epochs_checkpoint.every_n_epochs,
                save_top_k=cfg.train.every_n_epochs_checkpoint.save_top_k,
                verbose=cfg.train.every_n_epochs_checkpoint.verbose,
                save_last=cfg.train.every_n_epochs_checkpoint.save_last,
            )
        )

    return callbacks


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    glob_cfg = cfg
    cfg = glob_cfg.flow_train
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    
  
    dataset_config = glob_cfg["dataset"]
    glob_cfg["train"]["batch_size"] = glob_cfg["flow_train"]["batch_size"]["train"]
    hydra.utils.log.info(f"Instantiating <{dataset_config.name} dataset>")
    if dataset_config["name"] in ['sbm', 'comm20', 'planar', 'tree', 'ego_small']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, TreeSamplingMetrics, EgoSmallSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(glob_cfg)
        # pdb.set_trace()
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config['name'] == 'tree':
            sampling_metrics = TreeSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'ego_small':
            sampling_metrics = EgoSmallSamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if glob_cfg.model.type == 'HGVAE' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if glob_cfg.model.type == 'HGVAE' and glob_cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(glob_cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(glob_cfg)
            # datamodule.drop_last = True
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=glob_cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=glob_cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
            
            test_smiles = qm9_dataset.get_test_smiles(cfg=glob_cfg, test_dataloader=datamodule.test_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(glob_cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, glob_cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(glob_cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, glob_cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if glob_cfg.model.type == 'HGVAE' and glob_cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(glob_cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if glob_cfg.model.type == 'HGVAE':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(glob_cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    
    elif dataset_config["name"] in ['hyperbolic']:
        if dataset_config["name"] == 'hyperbolic':
            from datasets import hyp_dataset
            from analysis.spectre_utils import HyperbolicSamplingMetrics
            from analysis.visualization import NonMolecularVisualization
            
            datamodule = hyp_dataset.HyperbolicDataModule(glob_cfg)
            sampling_metrics = HyperbolicSamplingMetrics(datamodule)

            dataset_infos = hyp_dataset.HyperbolicDatasetInfos(datamodule, dataset_config)
            train_metrics = TrainAbstractMetricsDiscrete() if glob_cfg.model.type == 'HGVAE' else TrainAbstractMetrics()
            visualization_tools = NonMolecularVisualization()

            if glob_cfg.model.type == 'HGVAE' and glob_cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(glob_cfg.model.extra_features, dataset_info=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

            # # dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
            #                                         domain_features=domain_features)

            model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                            'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                            'extra_features': extra_features, 'domain_features': domain_features}


    else:
        raise NotImplementedError("Unknown dataset {}".format(glob_cfg["dataset"]))

    hydra_dir = Path.cwd()
    hydra.utils.log.info(f"Hydra Directory is {hydra_dir.resolve()}")

    get_model = ManifoldFMLitModule
    hydra.utils.log.info(f"Instantiating <{get_model}>")
    model = get_model(cfg, sampling_metrics, glob_cfg)

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    do_wandb_log = (WANDB_MODE.lower() != "disabled") and ("wandb" in cfg.logging)
    if do_wandb_log:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )
        
    if dataset_config["name"] in ['hyperbolic']:
        pass
    else:
        if not glob_cfg.model.use_poincare:
            glob_cfg.model.lgcn_in_channels = dataset_infos.input_dims['X'] + 1
            glob_cfg.model.lgcn_in_edge_channels = dataset_infos.input_dims['E'] + 1
            glob_cfg.model.lgcn_out_channels = glob_cfg.model.latent_channels   # lgcn out channels == hyperformer in_channels
        else:
            glob_cfg.model.lgcn_in_channels = dataset_infos.input_dims['X'] + dataset_infos.input_dims['y']
            glob_cfg.model.lgcn_in_edge_channels = dataset_infos.input_dims['E']
            glob_cfg.model.lgcn_out_channels = glob_cfg.model.latent_channels   # lgcn out channels == hyperformer in_channels
        glob_cfg.model.edge_classes = dataset_infos.output_dims['E']  ## only for edge attributes
        glob_cfg.model.node_classes = dataset_infos.output_dims['X']
        if glob_cfg.train.hyper_model == 'HGCN':
            VAE_model = HGCN(glob_cfg, **model_kwargs)
        elif glob_cfg.train.hyper_model == 'HGVAE':
            VAE_model = HGVAE(glob_cfg, **model_kwargs)

        if cfg.VAE_checkpoint is not None:
            VAE_checkpoint = torch.load(cfg.VAE_checkpoint)
            state_dict = VAE_checkpoint["state_dict"]
            VAE_model.load_state_dict(state_dict, strict=False)
            
        model.load_VAE(VAE_model)
        # pdb.set_trace()
        # Store the YaML config separately into the wandb dir
        # yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
        # (hydra_dir / "hparams.yaml").write_text(yaml_conf)


    ckpt = glob_cfg.flow_train.Flow_checkpoint
    if ckpt == "None":
        ckpt = None
    if cfg.test_only and ckpt is not None:
        flow_checkpoint = torch.load(ckpt)    
        state_dict = flow_checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        # model.load_from_checkpoint(ckpt, cfg=cfg, sampling_metrics=sampling_metrics, glob_cfg=glob_cfg)
    
    # import copy
    # old_ckpt = ckpt
    # new_ckpt = f"{old_ckpt}_fixed.ckpt"

    # ckpt = torch.load(old_ckpt, map_location="cpu")
    # fixed = copy.deepcopy(ckpt)          # 别动原文件

    # n_missing, n_cast = 0, 0
    # for opt_idx, opt_state in enumerate(fixed["optimizer_states"]):
    #     for pid, state in opt_state["state"].items():
    #         if "step" not in state:
    #             state["step"] = torch.tensor(129880.)
    #             n_missing += 1
    #         elif not torch.is_tensor(state["step"]):
    #             state["step"] = torch.tensor(state["step"])
    #             n_cast += 1

    # print(f"added {n_missing}, cast {n_cast} step fields")
    # pdb.set_trace()
    # torch.save(fixed, new_ckpt)
        

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        # default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=glob_cfg.general.check_val_every_n_epochs_flow,
        # progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        **cfg.train.pl_trainer,
    )


    hydra.utils.log.info("Starting training!")
    
    
    # test sampling method
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # samples = model.sample(10)
    # pdb.set_trace()


    if not cfg.test_only:    
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

        if do_wandb_log:
            hydra.utils.log.info(
                "W&B is no longer watching <{cfg.logging.wandb_watch.log}>!"
            )
            wandb_logger.experiment.unwatch(model)

        hydra.utils.log.info("Starting testing!")
        ckpt_path = "last" if cfg.train.pl_trainer.fast_dev_run else "best"
        trainer.test(datamodule=datamodule, ckpt_path=ckpt_path)

    else:
        # trainer.fit(model=model, datamodule=datamodule)
        hydra.utils.log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)
    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()



if __name__ == '__main__':
    main()
