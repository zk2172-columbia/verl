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
            n: ids.index.to_series().sample(frac=1).astype(int)
            for n, ids in self.source_df.groupby('data_source')
        }
        self.source_names = list(self.source_ids.keys())
        self.n_sources = len(self.source_names)

        self.step = 0
        self.weights = np.ones(self.n_sources) / self.n_sources
        self.hists = {n: {k: [] for k in [
            'step', 'stage', 'nsamples',
            'scr_mean', 'scr_std', 'rwd_mean', 'rwd_std',
            'adv_mean', 'adv_std', 'ret_mean', 'ret_std',
            'weight',
        ]} for n in self.source_names}

        self.initial_scrs = {}

        self.budget = 2000
        self.warmup_ratio = 0.25
        self.warmup_oversampling = 2.

        self.source_nsamples = {
            n: int(self.budget * self.warmup_ratio / self.n_sources)
            for n in self.source_names
        }

        self.stage = int(self.warmup_ratio == 0)
        # stage 0 - warmup
        # stage 1 - collect
        # stage 2 - train
        self.post_collect_sampling = True

    def __len__(self) -> int:
        # return len(self.dataset)
        return self.budget

    def __iter__(self) -> Iterator[int]:
        """
        Return one index
        """
        if self.stage < 2:
            if self.stage == 0:
                yield from pd.concat([
                    self.source_ids[n].iloc[:self.source_nsamples[n]]
                    for n in self.source_names
                ]).sample(frac=self.warmup_oversampling, replace=self.warmup_oversampling > 1)

                self.stage += 1

            while sum(self.source_nsamples.values()) < self.budget:
                n = np.random.choice(self.source_names, p=self.weights)
                i = int(self.source_ids[n].iloc[
                    self.source_nsamples[n] % len(self.source_ids[n])
                ])
                self.source_nsamples[n] += 1
                yield i

            self.source_ids_collected = {
                n: self.source_ids[n].iloc[:self.source_nsamples[n]]
                for n in self.source_names
            }
            self.stage += 1

        else:
            if self.post_collect_sampling:
                indices = dict.fromkeys(self.source_names, 0)
                source_ids = {
                    n: self.source_ids_collected[n].sample(frac=1).repeat(int(self.budget / self.source_nsamples[n]))
                    for n in self.source_names
                }

                while sum(indices.values()) < self.budget:
                    n = np.random.choice(self.source_names, p=self.weights)
                    i = int(source_ids[n].iloc[
                        indices[n] % self.source_nsamples[n]
                    ])
                    indices[n] += 1
                    yield i

            else:
                yield from pd.concat(self.source_ids_collected.values()).sample(frac=1)

    def update(self, batch: DataProto, metrics: Dict) -> None:
        # logging
        self.step += 1
        for n in metrics['data/sources']:
            self.hists[n]['step'].append(self.step)
            self.hists[n]['stage'].append(self.stage)
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

        print(self.source_nsamples)
        print(self.stage, sum(self.source_nsamples.values()), self.budget)

    def register_initial_scores(self, metrics):
        for k, s in metrics.items():
            name = '/'.join(k.split('/')[1:-2])
            # assert name in self.source_names
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
    def __init__(self, data_source, data_config):
        super().__init__(data_source, data_config)

        self.adv_cum = np.zeros_like(self.weights)
        self.eps = 1e-3

        # 1.5b - baseline
        self.benchmarks = {
            # 'DigitalLearningGmbH/MATH-lighteval/Algebra':                   0.763269,
            'DigitalLearningGmbH/MATH-lighteval/Counting & Probability':    0.463002,
            'DigitalLearningGmbH/MATH-lighteval/Geometry':                  0.437238,
            'DigitalLearningGmbH/MATH-lighteval/Intermediate Algebra':      0.346622,
            'DigitalLearningGmbH/MATH-lighteval/Number Theory':             0.492593,
            # 'DigitalLearningGmbH/MATH-lighteval/Prealgebra':                0.688863,
            'DigitalLearningGmbH/MATH-lighteval/Precalculus':               0.336996,
        }
        # 1.5b - random
        # self.benchmarks = {
        #     'DigitalLearningGmbH/MATH-lighteval/Algebra':                   0.731255,
        #     'DigitalLearningGmbH/MATH-lighteval/Counting & Probability':    0.452431,
        #     'DigitalLearningGmbH/MATH-lighteval/Geometry':                  0.418410,
        #     'DigitalLearningGmbH/MATH-lighteval/Intermediate Algebra':      0.294574,
        #     'DigitalLearningGmbH/MATH-lighteval/Number Theory':             0.455556,
        #     'DigitalLearningGmbH/MATH-lighteval/Prealgebra':                0.660161,
        #     'DigitalLearningGmbH/MATH-lighteval/Precalculus':               0.318681,
        # }
        # 4b
        # self.benchmarks = {
        #     'DigitalLearningGmbH/MATH-lighteval/Algebra':                   0.886729,
        #     'DigitalLearningGmbH/MATH-lighteval/Counting & Probability':    0.703625,
        #     'DigitalLearningGmbH/MATH-lighteval/Geometry':                  0.562105,
        #     'DigitalLearningGmbH/MATH-lighteval/Intermediate Algebra':      0.406912,
        #     'DigitalLearningGmbH/MATH-lighteval/Number Theory':             0.703704,
        #     'DigitalLearningGmbH/MATH-lighteval/Prealgebra':                0.853179,
        #     'DigitalLearningGmbH/MATH-lighteval/Precalculus':               0.418349,
        # }

    def _exp3(self, x, eta):
        e_x = np.exp(eta * (np.min(x) - x))
        return e_x / np.sum(e_x)

    def update(self, batch, metrics):
        super().update(batch, metrics)
        if self.stage < 1: return
        # adv = [h['adv_mean'][-1] for h in self.hists.values()]
        # adv = [h['reward_mean'][-1] for h in self.hists.values()]
        # adv = h['rwd_mean'][-1] - self.initial_scrs[k] for k, h in self.hists.items()]
        # self.adv_cum[k] += adv[k] / self.weights[k]
        ns = {
            n: self.hists[n]['nsamples'][-1] if self.hists[n]['step'][-1] == self.step else 0
            for n in self.source_names
        }
        adv = np.array([
            # h['scr_mean'][-1] - self.initial_scrs[k]
            self.hists[n]['scr_mean'][-1] - self.benchmarks[n] # * ns[n]
            if ns[n] > 0 else 0.
            for n in self.source_names
        ])
        adv_hat = adv / (np.sum([self.weights[k] for k, n in enumerate(self.source_names) if ns[n] > 0]) + self.eps)
        self.adv_cum += adv_hat
        # set p
        eta = np.sqrt(np.log(self.n_sources) / (self.n_sources * self.step))
        self.weights = self._exp3(self.adv_cum, eta=eta)
        print(adv)
        print(self.adv_cum)
        print(self.weights)


class Exp3DiffSampler(AdaptiveSampler):
    def __init__(self, data_source, data_config):
        super().__init__(data_source, data_config)

        self.adv_cum = np.zeros_like(self.weights)
        self.eps = 1e-3

    def _exp3(self, x, eta):
        e_x = np.exp(eta * (np.min(x) - x))
        return e_x / np.sum(e_x)

    def update(self, batch, metrics):
        super().update(batch, metrics)
        if self.stage < 1: return
        ns = {
            n: self.hists[n]['nsamples'][-1] if self.hists[n]['step'][-1] == self.step else 0
            for n in self.source_names
        }
        adv = np.array([
            self.hists[n]['scr_mean'][-1] - self.initial_scrs[n]
            # self.hists[n]['scr_mean'][-1] - self.benchmarks[n] # * ns[n]
            if ns[n] > 0 else 0.
            for n in self.source_names
        ])
        adv_hat = 2 * adv / (np.sum([self.weights[k] for k, n in enumerate(self.source_names) if ns[n] > 0]) + self.eps)
        self.adv_cum += adv_hat
        # set p
        eta = np.sqrt(np.log(self.n_sources) / (self.n_sources * self.step))
        self.weights = self._exp3(self.adv_cum, eta=eta)
        print(adv)
        print(self.adv_cum)
        print(self.weights)


class Exp3IXDiffSampler(AdaptiveSampler):
    def __init__(self, data_source, data_config):
        super().__init__(data_source, data_config)

        self.adv_cum = np.zeros_like(self.weights)
        self.eps = 2.

    def _eta(self):
        return np.sqrt(np.log(self.n_sources) / (self.n_sources * self.step))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def update(self, batch, metrics):
        super().update(batch, metrics)
        # if self.stage < 1: return
        ns = {
            n: self.hists[n]['nsamples'][-1] if self.hists[n]['step'][-1] == self.step else 0
            for n in self.source_names
        }
        adv = np.array([
            self.hists[n]['scr_mean'][-1] - self.initial_scrs[n]
            # self.hists[n]['scr_mean'][-1] - self.benchmarks[n] # * ns[n]
            if ns[n] > 0 else 0.
            for n in self.source_names
        ])
        # note that `+` has highe precedence than `if else`
        self.adv_cum += adv / (self.eps * self._eta() / 2 + (self.weights if self.stage > 0 else 0.))
        # set p
        self.weights = self._softmax(-self._eta() * self.adv_cum)
        print(adv)
        print(self.adv_cum)
        print(self.weights)


class IncExp3Sampler(AdaptiveSampler):
    def __init__(self, data_source, data_config):
        super().__init__(data_source, data_config)

        self.adv_cum = np.zeros_like(self.weights)
        self.adv_mov = np.zeros_like(self.weights)
        self.eps = 1e-3
        self.beta = 0.9

    def _exp3(self, x, eta):
        e_x = np.exp(eta * (np.min(x) - x))
        return e_x / np.sum(e_x)

    def update(self, batch, metrics):
        super().update(batch, metrics)
        if self.stage < 1: return
        adv = np.array([
            (self.hists[n]['scr_mean'][-1] - self.hists[n]['scr_mean'][-2]) / (self.hists[n]['step'][-1] - self.hists[n]['step'][-2])
            if len(self.hists[n]['scr_mean']) >= 2 else 0. #h['scr_mean'][-1] - self.initial_scrs[k]
            for n in self.source_names
        ]) / (self.weights + self.eps)
        self.adv_mov = self.beta * self.adv_mov + (1 - self.beta) * adv
        self.adv_cum += self.adv_mov / (1 - self.beta**self.step)
        # set p
        eta = np.sqrt(np.log(self.n_sources) / (self.n_sources * self.step))
        self.weights = self._exp3(-self.adv_cum, eta=eta)
        print(self.adv_mov)
        print(self.adv_cum)
        print(self.weights)
