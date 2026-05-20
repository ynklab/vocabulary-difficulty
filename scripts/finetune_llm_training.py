import inspect

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer
from torch.utils.data import Sampler


_TRAINER_INIT_PARAMS = inspect.signature(Trainer.__init__).parameters
_TRAINER_HAS_TOKENIZER_KWARG = 'tokenizer' in _TRAINER_INIT_PARAMS
_TRAINER_HAS_PROCESSING_CLASS_KWARG = 'processing_class' in _TRAINER_INIT_PARAMS


class HardnessCurriculumSampler(Sampler):
    def __init__(
        self,
        dataset_length,
        easy_indices,
        hard_indices,
        total_epochs,
        seed,
        ):
        self.dataset_length = int(dataset_length)
        self.easy_indices = [int(i) for i in easy_indices]
        self.hard_indices = [int(i) for i in hard_indices]
        self.total_epochs = int(total_epochs)
        self.seed = int(seed)
        self._iter_calls = 0
        if self.dataset_length <= 0:
            raise ValueError('dataset_length must be positive')
        if self.total_epochs <= 0:
            raise ValueError('total_epochs must be positive')

    def __len__(self):
        return self.dataset_length

    def _shuffled(self, values, rng):
        values = list(values)
        rng.shuffle(values)
        return values

    def _take(self, values, n):
        if n <= 0:
            return []
        if not values:
            return []
        return values[:n]

    def _fill_to_n(self, indices, rng):
        if len(indices) >= self.dataset_length:
            return indices[:self.dataset_length]
        fallback_pool = (
            self.easy_indices + self.hard_indices
            if (self.easy_indices or self.hard_indices)
            else list(range(self.dataset_length))
            )
        while len(indices) < self.dataset_length:
            fill = self._shuffled(fallback_pool, rng)
            if not fill:
                break
            need = self.dataset_length - len(indices)
            indices.extend(fill[:need])
        return indices[:self.dataset_length]

    def _epoch_one_indices(self, rng):
        if self.easy_indices:
            pass_1 = self._shuffled(self.easy_indices, rng)
            pass_2 = self._shuffled(self.easy_indices, rng)
            indices = (pass_1 + pass_2)[:self.dataset_length]
        else:
            # Fallback for tiny/degenerate split: no EASY bucket.
            pass_1 = self._shuffled(self.hard_indices, rng)
            pass_2 = self._shuffled(self.hard_indices, rng)
            indices = (pass_1 + pass_2)[:self.dataset_length]
        return self._fill_to_n(indices, rng)

    def _later_epoch_indices(self, rng):
        n = self.dataset_length
        easy_quota = int(round(0.2 * n))
        hard_quota = n - easy_quota

        easy_shuffled = self._shuffled(self.easy_indices, rng)
        easy_pick = self._take(easy_shuffled, min(easy_quota, len(easy_shuffled)))
        remaining_easy = easy_quota - len(easy_pick)

        hard_shuffled_1 = self._shuffled(self.hard_indices, rng)
        hard_pick = self._take(hard_shuffled_1, min(hard_quota, len(hard_shuffled_1)))
        remaining_hard = hard_quota - len(hard_pick)
        if remaining_hard > 0 and self.hard_indices:
            hard_shuffled_2 = self._shuffled(self.hard_indices, rng)
            hard_pick.extend(
                self._take(hard_shuffled_2, min(remaining_hard, len(hard_shuffled_2)))
                )
            remaining_hard = hard_quota - len(hard_pick)

        if remaining_easy > 0 and self.hard_indices:
            hard_fill_for_easy = self._shuffled(self.hard_indices, rng)
            easy_pick.extend(
                self._take(
                    hard_fill_for_easy,
                    min(remaining_easy, len(hard_fill_for_easy))
                    )
                )
        if remaining_hard > 0 and self.easy_indices:
            easy_fill_for_hard = self._shuffled(self.easy_indices, rng)
            hard_pick.extend(
                self._take(
                    easy_fill_for_hard,
                    min(remaining_hard, len(easy_fill_for_hard))
                    )
                )

        indices = easy_pick + hard_pick
        indices = self._fill_to_n(indices, rng)
        indices = self._shuffled(indices, rng)
        return indices

    def __iter__(self):
        epoch_idx = self._iter_calls
        self._iter_calls += 1
        if epoch_idx >= self.total_epochs:
            epoch_idx = self.total_epochs - 1
        rng = np.random.default_rng(self.seed + epoch_idx)
        if epoch_idx == 0:
            indices = self._epoch_one_indices(rng)
        else:
            indices = self._later_epoch_indices(rng)
        return iter(indices)


class CausalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [
            torch.tensor(f['input_ids'], dtype=torch.long)
            for f in features
            ]
        attention_mask = [
            torch.tensor(f['attention_mask'], dtype=torch.long)
            for f in features
            ]
        labels = [
            torch.tensor(f['labels'], dtype=torch.long)
            for f in features
            ]
        score_position = [
            torch.tensor(f.get('score_position', -1), dtype=torch.long)
            for f in features
            ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
            )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
            )
        score_position = torch.stack(score_position)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'score_position': score_position,
            }


class PromptOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [
            torch.tensor(f['input_ids'], dtype=torch.long)
            for f in features
            ]
        attention_mask = [
            torch.tensor(f['attention_mask'], dtype=torch.long)
            for f in features
            ]
        target_probs = [
            torch.tensor(f['target_probs'], dtype=torch.float32)
            for f in features
            ]
        sample_weight = [
            torch.tensor(f.get('sample_weight', 1.0), dtype=torch.float32)
            for f in features
            ]
        score_position = [
            torch.tensor(f.get('score_position', -1), dtype=torch.long)
            for f in features
            ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
            )
        target_probs = torch.stack(target_probs)
        sample_weight = torch.stack(sample_weight)
        score_position = torch.stack(score_position)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_probs': target_probs,
            'sample_weight': sample_weight,
            'score_position': score_position,
            }


def select_scoring_logits(logits, score_position=None):
    if score_position is None:
        return logits[:, -1, :]
    idx = score_position.to(device=logits.device, dtype=torch.long).view(-1)
    if idx.numel() != logits.size(0):
        raise ValueError(
            'score_position batch size mismatch: '
            f'{idx.numel()} vs {logits.size(0)}'
            )
    if torch.any(idx < 0) or torch.any(idx >= logits.size(1)):
        raise ValueError(
            'score_position contains out-of-range indices '
            f'for sequence length {logits.size(1)}'
            )
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_idx, idx, :]


class FormattedLossTrainer(Trainer):
    def __init__(
            self,
            *args,
            loss_decimals=6,
            curriculum_mode='none',
            curriculum_easy_indices=None,
            curriculum_hard_indices=None,
            curriculum_total_epochs=None,
            curriculum_seed=0,
            **kwargs
            ):
        trainer_kwargs = dict(kwargs)
        tokenizer = trainer_kwargs.get('tokenizer')

        # Transformers >=5 uses `processing_class` instead of `tokenizer`.
        if tokenizer is not None and not _TRAINER_HAS_TOKENIZER_KWARG:
            trainer_kwargs.pop('tokenizer', None)
            if (
                    _TRAINER_HAS_PROCESSING_CLASS_KWARG and
                    'processing_class' not in trainer_kwargs
                    ):
                trainer_kwargs['processing_class'] = tokenizer

        super().__init__(*args, **trainer_kwargs)
        if tokenizer is not None and not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        self.loss_decimals = loss_decimals
        self.curriculum_mode = curriculum_mode
        self.curriculum_easy_indices = list(curriculum_easy_indices or [])
        self.curriculum_hard_indices = list(curriculum_hard_indices or [])
        self.curriculum_total_epochs = curriculum_total_epochs
        self.curriculum_seed = int(curriculum_seed)

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        for key, value in logs.items():
            if key == 'loss' or key.endswith('_loss'):
                if isinstance(value, (float, np.floating)):
                    logs[key] = f'{float(value):.{self.loss_decimals}f}'
        super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None
                     ):
        if 'score_position' in inputs:
            inputs = dict(inputs)
            inputs.pop('score_position', None)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
            )

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = (
            self.train_dataset if train_dataset is None else train_dataset
            )
        if self.curriculum_mode != 'hardness2':
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                # Backward compatibility with older transformers.
                return super()._get_train_sampler()
        if train_dataset is None:
            raise ValueError('train_dataset is required for curriculum training')
        if self.curriculum_total_epochs is None:
            raise ValueError(
                'curriculum_total_epochs must be set for curriculum training'
                )
        return HardnessCurriculumSampler(
            dataset_length=len(train_dataset),
            easy_indices=self.curriculum_easy_indices,
            hard_indices=self.curriculum_hard_indices,
            total_epochs=self.curriculum_total_epochs,
            seed=self.curriculum_seed,
            )


class RaftLossTrainer(FormattedLossTrainer):
    def __init__(
            self,
            *args,
            class_token_ids=None,
            class_values=None,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        if class_token_ids is None:
            raise ValueError('class_token_ids must be provided for RAFT loss')
        if class_values is None:
            raise ValueError('class_values must be provided for RAFT loss')
        self.class_token_ids = class_token_ids
        self.class_values = class_values

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None
                     ):
        target_probs = inputs.pop('target_probs')
        sample_weight = inputs.pop('sample_weight', None)
        score_position = inputs.pop('score_position', None)
        outputs = model(**inputs)
        logits = outputs.logits
        next_logits = select_scoring_logits(logits, score_position=score_position)

        class_ids = torch.tensor(
            self.class_token_ids,
            dtype=torch.long,
            device=next_logits.device,
            )
        class_logits = next_logits.index_select(dim=1, index=class_ids)
        class_probs = F.softmax(class_logits.float(), dim=1)
        class_values = torch.tensor(
            self.class_values,
            dtype=torch.float32,
            device=next_logits.device,
            )
        pred_expected = (class_probs * class_values).sum(dim=1)
        target_expected = (target_probs.float() * class_values).sum(dim=1)
        per_item = ((pred_expected - target_expected) ** 2)
        if sample_weight is None:
            loss = per_item.mean()
        else:
            w = sample_weight.float()
            loss = (per_item * w).sum() / torch.clamp(w.sum(), min=1e-12)

        if return_outputs:
            return loss, outputs
        return loss


class SoftCELossTrainer(FormattedLossTrainer):
    def __init__(
            self,
            *args,
            class_token_ids=None,
            class_values=None,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        if class_token_ids is None:
            raise ValueError('class_token_ids must be provided for ce_prob loss')
        if class_values is None:
            raise ValueError('class_values must be provided for ce_prob loss')
        self.class_token_ids = class_token_ids
        self.class_values = class_values

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None
                     ):
        target_probs = inputs.pop('target_probs')
        sample_weight = inputs.pop('sample_weight', None)
        score_position = inputs.pop('score_position', None)
        outputs = model(**inputs)
        logits = outputs.logits
        next_logits = select_scoring_logits(logits, score_position=score_position)

        class_ids = torch.tensor(
            self.class_token_ids,
            dtype=torch.long,
            device=next_logits.device,
            )
        class_logits = next_logits.index_select(dim=1, index=class_ids)
        log_probs = F.log_softmax(class_logits.float(), dim=1)
        per_item = -(target_probs.float() * log_probs).sum(dim=1)
        if sample_weight is None:
            loss = per_item.mean()
        else:
            w = sample_weight.float()
            loss = (per_item * w).sum() / torch.clamp(w.sum(), min=1e-12)

        if return_outputs:
            return loss, outputs
        return loss


class DeltaSquaredLossTrainer(FormattedLossTrainer):
    def __init__(
            self,
            *args,
            class_token_ids=None,
            class_values=None,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        if class_token_ids is None:
            raise ValueError('class_token_ids must be provided for RAFT loss')
        if class_values is None:
            raise ValueError('class_values must be provided for RAFT loss')
        self.class_token_ids = class_token_ids
        self.class_values = class_values

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None
                     ):
        target_probs = inputs.pop('target_probs')
        sample_weight = inputs.pop('sample_weight', None)
        score_position = inputs.pop('score_position', None)
        outputs = model(**inputs)
        logits = outputs.logits
        next_logits = select_scoring_logits(logits, score_position=score_position)

        class_ids = torch.tensor(
            self.class_token_ids,
            dtype=torch.long,
            device=next_logits.device,
            )
        class_logits = next_logits.index_select(dim=1, index=class_ids)
        class_probs = F.softmax(class_logits.float(), dim=1)
        class_values = torch.tensor(
            self.class_values,
            dtype=torch.float32,
            device=next_logits.device,
            )
        target_expected = (target_probs.float() * class_values).sum(dim=1)

        per_item = ((
            (class_probs - target_probs.float()) *
            (class_values.unsqueeze(0) - target_expected.unsqueeze(1))
            ) ** 2).sum(dim=1)
        if sample_weight is None:
            loss = per_item.mean()
        else:
            w = sample_weight.float()
            loss = (per_item * w).sum() / torch.clamp(w.sum(), min=1e-12)

        if return_outputs:
            return loss, outputs
        return loss
