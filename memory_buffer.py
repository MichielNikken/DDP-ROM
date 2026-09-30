import numpy as np


class FlowReplayBuffer(object):
    def __init__(self, input_dim, mu_dim, timestep, size, batch_size=32):
        self.input_dim = input_dim
        self.timestep = timestep
        self.size = int(size*timestep)
        self.vx_buf = np.zeros([int(self.size), int(input_dim)], dtype=np.float32)
        self.vy_buf = np.zeros([int(self.size), int(input_dim)], dtype=np.float32)
        self.mus_buf = np.zeros([int(self.size), int(mu_dim)], dtype=np.float32)
        self.dt_buf = np.zeros((int(self.size),), dtype=np.float32)
        self.batch_size = batch_size
        self.ptr, self.size, self.max_size = 0, 0, int(size*timestep)
        self.idx = 0

    def store(self, vx, vy, mu, dt):
        self.vx_buf[self.ptr] = vx
        self.vy_buf[self.ptr] = vy
        self.mus_buf[self.ptr] = mu
        self.dt_buf[self.ptr] = dt
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        for i in range(self.batch_size):
            if idxs[i] >= self.size - 1 or self.dt_buf[idxs[i] + 1] == 0.0:
                idxs[i] = idxs[i] - 1

        return dict(vx_snapshots=self.vx_buf[idxs],
                    vy_snapshots=self.vy_buf[idxs],
                    vx_snapshots_next=self.vx_buf[idxs + 1],
                    vy_snapshots_next=self.vy_buf[idxs + 1],
                    mus=self.mus_buf[idxs],
                    dts=self.dt_buf[idxs])

    def get_all_samples(self):
        return dict(vx_snapshots=self.vx_buf,
                    vy_snapshots=self.vy_buf,
                    mus=self.mus_buf,
                    dts=self.dt_buf)

    def __iter__(self):
        self.order = np.random.permutation(self.size).tolist()
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.size - self.batch_size:
            idxs = self.order[self.idx:self.idx + self.batch_size]

        elif self.idx < self.size:
            idxs = self.order[self.idx:]
        else:
            raise StopIteration

        idxs2 = []
        for i in range(self.batch_size):
            if idxs[i] >= self.size - 1 or self.dt_buf[idxs[i] + 1] == 0.0:
                idxs[i] = idxs[i] - 1
            idxs2.append(idxs[i] + 1)
        batch = dict(vx_snapshots=self.vx_buf[idxs],
                     vy_snapshots=self.vy_buf[idxs],
                     vx_snapshots_next=self.vx_buf[idxs2],
                     vy_snapshots_next=self.vy_buf[idxs2],
                     mus=self.mus_buf[idxs],
                     dts=self.dt_buf[idxs])
        self.idx += self.batch_size
        return idxs, batch


class ReplayBuffer(object):
    def __init__(self, input_dim, mu_dim, timestep, size, batch_size=32):
        self.input_dim = input_dim
        self.timestep = timestep
        self.size = int(size*timestep)
        self.obs_buf = np.zeros([int(self.size), int(input_dim)], dtype=np.float32)
        self.mus_buf = np.zeros([int(self.size), int(mu_dim)], dtype=np.float32)
        self.dt_buf = np.zeros((int(self.size),), dtype=np.float32)
        self.batch_size = batch_size
        self.ptr, self.size, self.max_size = 0, 0, int(size*timestep)
        self.idx = 0

    def store(self, obs, mu, dt):
        self.obs_buf[self.ptr] = obs
        self.mus_buf[self.ptr] = mu
        self.dt_buf[self.ptr] = dt
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        for i in range(self.batch_size):
            if idxs[i] >= self.size - 1:
                idxs[i] = idxs[i] - 1

        return dict(snapshots=self.obs_buf[idxs],
                    mus=self.mus_buf[idxs],
                    dts=self.dt_buf[idxs])

    def get_all_samples(self):
        return dict(snapshots=self.obs_buf,
                    mus=self.mus_buf,
                    dts=self.dt_buf)

    def __iter__(self):
        self.order = np.random.permutation(self.size).tolist()
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.size - self.batch_size:
            idxs = self.order[self.idx:self.idx + self.batch_size]

        elif self.idx < self.size:
            idxs = self.order[self.idx:]
        else:
            raise StopIteration

        batch = dict(snapshots=self.obs_buf[idxs],
                     mus=self.mus_buf[idxs],
                     dts=self.dt_buf[idxs])
        self.idx += self.batch_size
        return idxs, batch