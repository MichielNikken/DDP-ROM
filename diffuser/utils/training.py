import os
import copy
import pickle
from datetime import datetime

import torch

from .arrays import batch_to_device
from .timer import Timer

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA:
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        val_dataset=None,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        save_freq=1000,
        bucket=None,
        device='cuda',
        save_checkpoints=False,
    ):
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        self.save_checkpoints = save_checkpoints
        self.log_freq = log_freq
        self.save_freq = save_freq

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.val_dataset = val_dataset

        self.dataloader = cycle(
            torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
            )
        )

        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket

        self.reset_parameters()
        self.step = 0

        self.device = device
        self.init_dt = datetime.now().strftime('%Y%m%d_%H%M%S')

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, patience=1e6, logs=None):
        timer = Timer()
        logs = {} if logs is None else logs
        self.model.train()
        if 'best_val_loss' not in logs:
            logs['best_val_loss'] = float('inf')
        current_patience = patience
        for step in range(n_train_steps):
            if current_patience == 0:
                break
            loss_acc = 0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                loss_acc += loss

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                if self.save_checkpoints:
                    self.save(name=f'state_{self.step}')
                else:
                    self.save(name=f'state')

            if self.step % self.log_freq == 0:
                if self.val_dataset is not None:
                    val_loss, val_loss_ema = self.validate()
                    if val_loss < logs['best_val_loss']:
                        current_patience = patience
                        self.save(name=f'best_val_{self.init_dt}')
                        logs['best_val_loss'] = val_loss
                    else:
                        current_patience -= 1


                    infos['val_loss'] = val_loss
                    infos['val_loss_ema'] = val_loss_ema
                    self.model.train()
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'step: {self.step}| loss: {loss_acc:8.4f} | {infos_str} | patience: {current_patience} | t: {timer():8.4f}')

                # Detach gradients and convert to float
                metrics = {k: v.detach().item() if isinstance(v, torch.Tensor) else v for k, v in infos.items()}
                metrics['train_loss'] = loss_acc.detach().item()
                metrics['step'] = self.step
                for key, val in metrics.items():
                    if key not in logs:
                        logs[key] = []
                    logs[key].append(val)
                with open(os.path.join(self.bucket, 'logs_'+ self.init_dt + '.pickle'), 'wb') as f:
                    pickle.dump(logs, f)
            self.step += 1
        return logs

    def validate(self):
        valdata = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True
        )
        n_samples = len(self.val_dataset)
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            loss_model = 0
            loss_ema = 0
            for batch in valdata:
                batch = batch_to_device(batch, device=self.device)
                w = len(batch.trajectories)/ n_samples # batches may differ in size
                loss_model += self.model.loss(*batch)[0] * w
                loss_ema += self.ema_model.loss(*batch)[0] * w
        return torch.mean(loss_model), torch.mean(loss_ema)


    def save(self, name='state'):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.join(savepath, f'{name}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, f'checkpoint/state.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
