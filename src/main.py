import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.HGCN import HGCN
from src.HGVAE import HGVAE
import pdb
from utils import setup_wandb_logger

warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    
    seed_everything(cfg.train.seed)
    
    if dataset_config["name"] in ['sbm', 'comm20', 'planar', 'tree', 'ego_small']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, TreeSamplingMetrics, EgoSmallSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
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
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'HGVAE' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'HGVAE' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses', 'zinc']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        elif dataset_config.name == 'zinc':
            from datasets import zinc_dataset
            datamodule = zinc_dataset.ZINCDataModule(cfg)
            dataset_infos = zinc_dataset.ZINCinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'HGVAE' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'HGVAE':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))


    utils.create_folders(cfg)
    
    # X, E, y = dataset_infos.get_example()
    # HGCN input adjacent matrix(ABSTRACT DATASET). BUT MOLECULES CONTAIN EDGE FEATURE.
    if not cfg.model.use_poincare:
        cfg.model.lgcn_in_channels = dataset_infos.input_dims['X'] + 1
        cfg.model.lgcn_in_edge_channels = dataset_infos.input_dims['E'] + 1
        cfg.model.lgcn_out_channels = cfg.model.latent_channels   # lgcn out channels == hyperformer in_channels
    else:
        # pdb.set_trace()
        cfg.model.lgcn_in_channels = dataset_infos.input_dims['X'] + dataset_infos.input_dims['y']
        cfg.model.lgcn_in_edge_channels = dataset_infos.input_dims['E']
        cfg.model.lgcn_out_channels = cfg.model.latent_channels   # lgcn out channels == hyperformer in_channels
    cfg.model.edge_classes = dataset_infos.output_dims['E']  ## only for edge attributes
    cfg.model.node_classes = dataset_infos.output_dims['X']
    
    if cfg.train.hyper_model == 'HGCN':
        hyper_model = HGCN(cfg, **model_kwargs)
    elif cfg.train.hyper_model == 'HGVAE':
        hyper_model = HGVAE(cfg, **model_kwargs)
    

    # hyper_model.load_btc_model("/butianci/HypeFlow/src/lorentz_to_poincare_model.pth")
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                                filename='{epoch}-{batch_loss:.2f}',
                                                monitor='val_epoch/log_metric',
                                                save_top_k=5,
                                                mode='max',
                                                every_n_epochs=5)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    wandb_logger = setup_wandb_logger(cfg)
    hyper_trainer = Trainer( # gradient_clip_val=cfg.train.clip_grad,
                        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                        accelerator='gpu' if use_gpu else 'cpu',
                        devices=cfg.general.gpus if use_gpu else 1,
                        max_epochs=cfg.train.n_epochs,
                        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                        fast_dev_run=cfg.general.name == 'debug',
                        enable_progress_bar=True,
                        callbacks=callbacks,
                        log_every_n_steps=50 if name != 'debug' else 1,
                        logger = wandb_logger)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyper_model = hyper_model.to(device)
    
    # samples = hyper_model.sample_batch(20)
    

    if not cfg.general.test_only:
        hyper_trainer.fit(hyper_model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            hyper_trainer.test(hyper_model, datamodule=datamodule)
    
    # trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
    #                   strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
    #                   accelerator='gpu' if use_gpu else 'cpu',
    #                   devices=cfg.general.gpus if use_gpu else 1,
    #                   max_epochs=cfg.train.n_epochs,
    #                   check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
    #                   fast_dev_run=cfg.general.name == 'debug',
    #                   enable_progress_bar=False,
    #                   callbacks=callbacks,
    #                   log_every_n_steps=50 if name != 'debug' else 1,
    #                   logger = [])

    # if not cfg.general.test_only:
    #     trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    #     if cfg.general.name not in ['debug', 'test']:
    #         trainer.test(model, datamodule=datamodule)
    # else:
    #     # Start by evaluating test_only_path
    #     trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
    #     if cfg.general.evaluate_all_checkpoints:
    #         directory = pathlib.Path(cfg.general.test_only).parents[0]
    #         print("Directory:", directory)
    #         files_list = os.listdir(directory)
    #         for file in files_list:
    #             if '.ckpt' in file:
    #                 ckpt_path = os.path.join(directory, file)
    #                 if ckpt_path == cfg.general.test_only:
    #                     continue
    #                 print("Loading checkpoint", ckpt_path)
    #                 trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
