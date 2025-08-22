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

        self.step = 0
        self.weights = np.ones(len(self.source_names)) / len(self.source_names)
        self.hists = {n: {k: [] for k in [
            'step', 'nsamples',
            'adv_mean', 'adv_std', 'reward_mean', 'reward_std',
            'weight',
        ]} for n in self.source_names}

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """
        Return one index
        """
        source_ids = [ids.sample(frac=1).values for ids in self.source_ids.values()]
        n_samples_all = np.array([len(ids) for ids in source_ids])
        n_sources = len(source_ids)

        indices = np.zeros_like(n_samples_all, dtype=int)

        while (indices < n_samples_all).all():
            k = np.random.choice(n_sources, p=self.weights)
            i = int(source_ids[k][indices[k]])
            indices[k] += 1
            yield i

    def update(self, batch: DataProto, metrics: Dict) -> None:
        # logging
        self.step += 1
        for n in metrics['data/sources']:
            self.hists[n]['step'].append(self.step)
            self.hists[n]['nsamples'].append(metrics['data/nsamples/sources'][n])
            self.hists[n]['adv_mean'].append(metrics['critic/advantages/sources/mean'][n])
            self.hists[n]['adv_std'].append(metrics['critic/advantages/sources/std'][n])
            self.hists[n]['reward_mean'].append(metrics['critic/rewards/sources/mean'][n])
            self.hists[n]['reward_std'].append(metrics['critic/rewards/sources/std'][n])
            self.hists[n]['weight'].append(self.weights[self.source_names.index(n)])


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
        k = np.argmin(adv)
        # set p
        pmin = 0.2
        self.weights.fill(pmin)
        self.weights[k] = 1. - pmin * (len(self.weights) - 1)


# class Exp3Sampler(AbstractSampler):
#     """
#     BatchSampler that works with Exp3Sampler, inheriting from UCBBatchSampler.
#     Only __init__ and __iter__ are overridden to swap UCB metrics for Exp3.
#     """
#     def __init__(
#         self,
#         data_source: RLHFDataset,
#         data_config: DictConfig,
#     ):
#         super().__init__(data_source=data_source, data_config=data_config)
#         self.dataset = data_source

#         self.batch_size = data_config.train_batch_size
#         self.gamma = 0.1

#         self.sources = self.dataset.dataframe.select_columns('data_source').to_pandas()
#         breakpoint()
#         self.source_names = np.unique(self.sources)
#         self.n_sources = len(self.source_names)

#     def __len__(self) -> int:
#         return len(self.dataset)

#     def __iter__(self) -> Iterator[List[int]]:
#         # Same bucketing logic as UCBBatchSampler
#         buckets = {src: [] for src in self.source_names}
#         for idx in range(len(self.sources)):
#             src = self.sources[idx]
#             buckets[src].append(idx)
#         for b in buckets:
#             random.shuffle(b)

#         breakpoint()

#         while True:
#             probs = self.ucb_sampler.select_source()
#             batch = []
#             cnt_per_src = [0] * self.n_sources
#             attempts = 0
#             while len(batch) < self.batch_size and attempts < self.batch_size * 10:
#                 s = random.choices(
#                     range(self.n_sources),
#                     weights=probs,
#                     k=1
#                 )[0]
#                 if buckets[s]:
#                     batch.append(buckets[s].pop())
#                     cnt_per_src[s] += 1
#                 attempts += 1

#             if len(batch) == self.batch_size or (not self.drop_last and batch):
#                 self.ucb_sampler.src_cnt = cnt_per_src
#                 # Print Exp3-specific metrics including counts and inherited stats
#                 print(
#                     f"Exp3 t={self.ucb_sampler.t}, "
#                     f"R={self.ucb_sampler.R}, "
#                     f"N={self.ucb_sampler.N}, "
#                     f"w={self.ucb_sampler.w}, "
#                     f"probs={probs}, "
#                     f"cnt={cnt_per_src}"
#                 )
#                 yield batch
#             else:
#                 break

#     def update(self, batch):
#         return self.ucb_sampler.update(probs, metrics)

#     @property
#     def last_probs(self) -> List[float]:
#         return self.ucb_sampler.last_probs
