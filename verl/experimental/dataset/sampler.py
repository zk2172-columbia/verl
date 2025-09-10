# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod
from collections.abc import Sized

from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset


from typing import List, Dict, Optional, Iterator, Sized

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass


class AdaptiveSampler(AbstractSampler):
    """
    Base class for adaptive sampler
    """
    def __init__(
        self,
        data_source: RLHFDataset,
        data_config: DictConfig,
    ):
        super().__init__(data_source=data_source, data_config=data_config)
        self.dataset = data_source
        self.data_config = data_config

        self.source_df = self.dataset.dataframe.select_columns('data_source').to_pandas()
        self.source_ids = {
            n: ids.index.to_series()
            for n, ids in self.source_df.groupby('data_source')
        }
        self.source_names = list(self.source_ids.keys())
        self.n_sources = len(self.source_names)

        self.step = 0
        self.weights = np.ones(self.n_sources) / self.n_sources
        self.hists = {n: {k: [] for k in [
            'step', 'nsamples',
            'scr_mean', 'scr_std', 'rwd_mean', 'rwd_std',
            'adv_mean', 'adv_std', 'ret_mean', 'ret_std',
            'weight',
        ]} for n in self.source_names}

        self.initial_scrs = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """
        Return one index
        """
        source_ids = [ids.sample(frac=1).values for ids in self.source_ids.values()]
        n_samples_all = np.array([len(ids) for ids in source_ids])

        indices = np.zeros_like(n_samples_all, dtype=int)

        while (indices < n_samples_all).all():
            k = np.random.choice(self.n_sources, p=self.weights)
            i = int(source_ids[k][indices[k]])
            indices[k] += 1
            yield i

    def update(self, batch: DataProto, metrics: Dict) -> None:
        # logging
        self.step += 1
        for n in metrics['data/sources']:
            self.hists[n]['step'].append(self.step)
            self.hists[n]['nsamples'].append(metrics['data/nsamples/sources'][n])
            self.hists[n]['scr_mean'].append(metrics['critic/score/sources/mean'][n])
            self.hists[n]['scr_std'].append(metrics['critic/score/sources/std'][n])
            self.hists[n]['rwd_mean'].append(metrics['critic/rewards/sources/mean'][n])
            self.hists[n]['rwd_std'].append(metrics['critic/rewards/sources/std'][n])
            self.hists[n]['adv_mean'].append(metrics['critic/advantages/sources/mean'][n])
            self.hists[n]['adv_std'].append(metrics['critic/advantages/sources/std'][n])
            self.hists[n]['ret_mean'].append(metrics['critic/returns/sources/mean'][n])
            self.hists[n]['ret_std'].append(metrics['critic/returns/sources/std'][n])
            self.hists[n]['weight'].append(self.weights[self.source_names.index(n)])

    def register_initial_scores(self, metrics):
        for k, s in metrics.items():
            name = '/'.join(k.split('/')[1:-2])
            assert name in self.source_names
            self.initial_scrs[name] = s


class RandomSampler(AdaptiveSampler):
    """
    Random sampler
    No changes required
    """
    def __init__(self, data_source, data_config):
        super().__init__(data_source, data_config)


class GreedySampler(AdaptiveSampler):
    """
    Greedy sampler
    Implement the update method
    """
    def update(self, batch, metrics):
        super().update(batch, metrics)
        # select based on lastest minimum adv
        # adv = [h['adv_mean'][-1] for h in self.hists.values()]
        adv = [h['reward_mean'][-1] for h in self.hists.values()]
        # set p
        # k = np.argmin(adv)
        self.weights[0] = 0.
        self.weights[1] = 1.
        # pmin = 0.2
        # self.weights.fill(pmin)
        # self.weights[k] = 1. - pmin * (len(self.weights) - 1)


class Exp3Sampler(AdaptiveSampler):
    """
    Greedy sampler
    Implement the update method
    """

    def __init__(self, data_source, data_config):
        super().__init__(data_source, data_config)

        self.adv_cum = np.zeros_like(self.weights)
        self.eps = 1e-3

    def _exp3(self, x, eta):
        e_x = np.exp(-eta * (x - np.max(x)))
        return e_x / np.sum(e_x)

    def update(self, batch, metrics):
        super().update(batch, metrics)
        # adv = [h['adv_mean'][-1] for h in self.hists.values()]
        # adv = [h['reward_mean'][-1] for h in self.hists.values()]
        # adv = h['rwd_mean'][-1] - self.initial_scrs[k] for k, h in self.hists.items()]
        # self.adv_cum[k] += adv[k] / self.weights[k]
        adv = np.array([
            h['scr_mean'][-1] - self.initial_scrs[k]
            if h['step'][-1] == self.step else 0.
            for k, h in self.hists.items()
        ])
        self.adv_cum += adv / (self.weights + self.eps)
        # set p
        eta = np.sqrt(np.log(self.n_sources) / (self.n_sources * self.step))
        self.weights = self._exp3(self.adv_cum, eta=eta)
        print(self.adv_cum)
        print(self.weights)


class ExpSampler(AdaptiveSampler):
    """
    Greedy sampler
    Implement the update method
    """

    def __init__(self, data_source, data_config):
        super().__init__(data_source, data_config)

        self.adv_cum = np.zeros_like(self.weights)
        self.adv_mov = np.zeros_like(self.weights)
        self.eps = 1e-3
        self.beta = 0.9

    def _exp3(self, x, eta):
        e_x = np.exp(-eta * (x - np.max(x)))
        return e_x / np.sum(e_x)

    def update(self, batch, metrics):
        super().update(batch, metrics)
        adv = np.array([
            (h['scr_mean'][-1] - h['scr_mean'][-2]) / (h['step'][-1] - h['step'][-2] + self.eps)
            if len(h['scr_mean']) >= 2 else 0. #h['scr_mean'][-1] - self.initial_scrs[k]
            for k, h in self.hists.items()
        ]) / (self.weights + self.eps)
        self.adv_mov = self.beta * self.adv_cum + (1 - self.beta) * adv
        self.adv_cum += self.adv_mov / (1 - self.beta**self.step)
        # set p
        eta = np.sqrt(np.log(self.n_sources) / (self.n_sources * self.step))
        self.weights = self._exp3(self.adv_cum, eta=eta)
        print(self.adv_mov)
        print(self.adv_cum)
        print(self.weights)
