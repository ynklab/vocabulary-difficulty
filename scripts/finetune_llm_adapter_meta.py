import json
from pathlib import Path

import numpy as np


ADAPTER_METADATA_FILENAME = 'adapter_metadata.json'


def adapter_metadata_path(adapter_path):
    return Path(adapter_path) / ADAPTER_METADATA_FILENAME


def load_adapter_metadata(adapter_path):
    meta_path = adapter_metadata_path(adapter_path)
    if not meta_path.exists():
        return None
    with meta_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_adapter_metadata(adapter_path, metadata):
    adapter_path = Path(adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    meta_path = adapter_metadata_path(adapter_path)
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write('\n')


def parse_adapter_calibration(metadata):
    if not metadata:
        return None
    calib = metadata.get('calibration')
    if calib is None:
        # Backward compatibility with previously saved adapters.
        calib = metadata.get('linear_correction')
    if not isinstance(calib, dict):
        return None
    if not calib.get('enabled', False):
        return None
    try:
        coef = float(calib['coef'])
        intercept = float(calib['intercept'])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        'coef': coef,
        'intercept': intercept,
        }


def build_adapter_metadata(
    scale,
    active_space,
    data_min,
    data_max,
    bin_edges_active,
    prob_class_values_active,
    calibration=None,
    model_family='causal',
    metadata_version=2,
    ):
    calib_meta = {'enabled': False}
    if calibration is not None:
        calib_meta = {
            'enabled': True,
            'coef': float(calibration['coef']),
            'intercept': float(calibration['intercept']),
            }
    return {
        'metadata_version': int(metadata_version),
        'scale_min': int(scale.min),
        'scale_max': int(scale.max),
        'space': str(active_space),
        'model_family': str(model_family),
        'data_min': float(data_min),
        'data_max': float(data_max),
        'bin_edges_active': [
            float(x) for x in np.asarray(bin_edges_active, dtype=float)
            ],
        'prob_class_values_active': [
            float(x) for x in np.asarray(prob_class_values_active, dtype=float)
            ],
        'calibration': calib_meta,
        }


def get_metadata_array(metadata, key, expected_shape):
    if not metadata or (key not in metadata):
        return None, None
    arr = np.asarray(metadata[key], dtype=float)
    if tuple(arr.shape) != tuple(expected_shape):
        return None, (
            f'ignoring adapter metadata {key} due to shape mismatch '
            f'{arr.shape} != {expected_shape}'
            )
    return arr, None


def resolve_effective_scale_and_space(
    metadata,
    mode,
    cli_scale_min,
    cli_scale_max,
    cli_space,
    ):
    messages = []
    scale_min = int(cli_scale_min)
    scale_max = int(cli_scale_max)
    active_space = str(cli_space)

    if metadata:
        active_space = str(metadata.get('space', cli_space))
        if mode == 'predict' and active_space != cli_space:
            messages.append(
                'Adapter metadata overrides active space: '
                f'{cli_space} -> {active_space}'
                )

    if mode == 'predict' and metadata:
        meta_scale_min = metadata.get('scale_min')
        meta_scale_max = metadata.get('scale_max')
        if (
            isinstance(meta_scale_min, int) and
            isinstance(meta_scale_max, int) and
            (meta_scale_max > meta_scale_min)
            ):
            if (
                (meta_scale_min != cli_scale_min) or
                (meta_scale_max != cli_scale_max)
                ):
                messages.append(
                    'Adapter metadata overrides scale range: '
                    f'{cli_scale_min}..{cli_scale_max} -> '
                    f'{meta_scale_min}..{meta_scale_max}'
                    )
            scale_min = meta_scale_min
            scale_max = meta_scale_max

    return {
        'scale_min': scale_min,
        'scale_max': scale_max,
        'active_space': active_space,
        'messages': messages,
        }


def resolve_saved_interpretation_params(metadata, n_points):
    messages = []
    saved_edges, saved_edges_err = get_metadata_array(
        metadata,
        'bin_edges_active',
        expected_shape=(n_points + 1,),
        )
    if saved_edges_err:
        messages.append(saved_edges_err)

    saved_prob_class_values, saved_prob_class_values_err = get_metadata_array(
        metadata,
        'prob_class_values_active',
        expected_shape=(n_points,),
        )
    if saved_prob_class_values_err:
        messages.append(saved_prob_class_values_err)

    data_min = None
    data_max = None
    if metadata:
        raw_min = metadata.get('data_min')
        raw_max = metadata.get('data_max')
        if (raw_min is not None) and (raw_max is not None):
            try:
                data_min = float(raw_min)
                data_max = float(raw_max)
            except (TypeError, ValueError):
                messages.append(
                    'ignoring adapter metadata data_min/data_max due to '
                    'invalid numeric values'
                    )
                data_min = None
                data_max = None

    return {
        'bin_edges_active': saved_edges,
        'prob_class_values_active': saved_prob_class_values,
        'data_min': data_min,
        'data_max': data_max,
        'messages': messages,
        }
