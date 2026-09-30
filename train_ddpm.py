import numpy as np
import torch

from config.config import Config
from diffuser import utils
from diffuser.datasets import SequenceDataset
from diffuser.models import GaussianDiffusion, TemporalUnet
from diffuser.utils import Trainer
from models import AEMLP, ConvAE, AE_INN
from utils import transform


def main():
    print("Configuring DDPM...")
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    data_train = SequenceDataset(
        Config.train_file,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_path_length=Config.max_path_length,
        max_n_episodes=Config.max_n_episodes,
        termination_penalty=0,
        use_padding=Config.use_padding,
        discount=Config.discount,
        param_scale=Config.param_scale,
        include_params=Config.include_params,
    )

    data_val = SequenceDataset(
        Config.val_file,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_path_length=Config.max_path_length,
        max_n_episodes=Config.max_n_episodes,
        termination_penalty=0,
        use_padding=Config.use_padding,
        discount=Config.discount,
        param_scale=Config.param_scale,
        include_params=Config.include_params,
    ) if Config.val_file is not None else None

    # Transforms corresponding to data
    data = np.load(Config.train_file, allow_pickle=True)
    ae_type = Config.ae_type
    if ae_type == 'mlp':
        ae = AEMLP.load_config(Config.ae_path)
    elif ae_type == 'conv':
        ae = ConvAE.load_config(Config.ae_path)
    elif ae_type == 'inn':
        ae = AE_INN.load_config(Config.ae_path)
    elif ae_type is None:
        ae = None
    else:
        raise ValueError('Unknown ae_type {}'.format(ae_type))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ae is not None:
        weights = torch.load(Config.ae_weights)
        ae.load_state_dict(weights['model'])
        print(f"Using device: {device}")
        ae.to(device)
    scaler = data['scaler'] if Config.loss_descale else None
    svd = data['svd'] if Config.loss_project_svd else None

    transform_before_loss = lambda *x: transform(*x, mode='decode', xy_axis=-1, scaler=scaler, svd=svd, ae=ae, device=device)

    model = TemporalUnet(
        Config.horizon,
        data_train.observation_dim,
        param_dim=Config.param_dim,
        dim=Config.dim,
        dim_mults= Config.dim_mults,
        param_condition=Config.param_condition,
        condition_dropout=Config.condition_dropout,
        calc_energy=Config.calc_energy,
        kernel_size=Config.kernel_size,
    )

    diffusion = GaussianDiffusion(
        model,
        Config.horizon,
        data_train.observation_dim,
        data_train.action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        action_weight=Config.action_weight,
        loss_discount=Config.loss_discount,
        loss_weights=Config.loss_weights,
        param_condition=Config.param_condition,
        condition_guidance_w=Config.condition_guidance_w,
        transform_before_loss=transform_before_loss
    )
    diffusion.to(device)

    trainer = Trainer(
        diffusion,
        data_train,
        val_dataset=data_val,
        ema_decay=Config.ema_decay,
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        step_start_ema=Config.step_start_ema,
        update_ema_every=Config.update_ema_every,
        log_freq=Config.log_freq,
        save_freq=Config.save_freq,
        bucket=Config.bucket,
        device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )

    utils.report_parameters(model)

    print('Testing forward/backward...', end=' ', flush=True)
    batch = utils.batchify(data_train[0], Config.device)
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    print('passed.')

    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)
    logs = {'config': Config.loggable()}
    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs}')
        logs = trainer.train(n_train_steps=Config.n_steps_per_epoch, patience=Config.patience, logs=logs)

if __name__ == '__main__':
    main()