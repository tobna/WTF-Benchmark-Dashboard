import contextlib
import json
import os
import sys
from json import JSONDecodeError
from time import sleep
import pandas as pd
import taxonomy as tx

_DATA_FILE = os.path.join('data', 'data.json')
_DATETIME_FORMAT = '%d.%m.%Y %H:%M:%S'
# ----------------------- INFO --------------------
_COLUMN_NAMES = {'run name': 'run_name', 'run date': 'run_date', 'model': 'model', 'GPU type (finetuning)': 'device',

                 # ----------------------- HYPER PARAMS --------------------
                 'epochs (pretraining)': 'pre-train_epochs', 'image resolution (pretraining) [px]': 'pre-train_imsize',
                 'GPUS (pretraining)': 'pre-train_world_size', 'lr (pretraining)': 'pre-train_lr',
                 'per-GPU batch size (pretraining) [ims]': 'pre-train_local_batch_size',
                 'dataloader workers (pretraining)': 'pre-train_num_workers',
                 'batch size (pretraining) [ims]': 'pre-train_batch_size',
                 'dataset (pretraining)': 'pre-train_dataset',

                 'image resolution (finetuning) [px]': 'imsize', 'epochs (finetuning)': 'final_epoch',
                 'GPUs (finetuning)': 'world_size', 'per-GPU batch size (finetuning) [ims]': 'local_batch_size',
                 'lr (finetuning)': 'lr', 'dataloader workers (finetuning)': 'num_workers',
                 'dataset (finetuning)': 'dataset', 'batch size (finetuning) [ims]': 'batch_size',

                 'pre-norm': 'pre_norm', 'shuffle': 'shuffle', 'layer scale init': 'pre-train_layer_scale_init_values',
                 'gaussian blur (augmentation)': 'aug_gauss_blur', 'amp (for evaluation)': 'eval_amp',
                 'dropout': 'dropout', 'color jitter factor (augmentation)': 'aug_color_jitter_factor',
                 'resizing (augmentation)': 'aug_resize', 'pin_memory': 'pin_memory', 'warmup schedule': 'warmup_sched',
                 'warmup epochs': 'warmup_epochs', 'augmentation strategy': 'augment_strategy',
                 'normalization (augmentation)': 'aug_normalize', 'label smoothing (augmentation)': 'label_smoothing',
                 'gradient norm clip (max)': 'max_grad_norm', 'cutmix (augmentation)': 'aug_cutmix',
                 'qkv bias': 'pre-train_qkv_bias', 'lr schedule': 'sched', 'optimizer': 'opt',
                 'solarization (augmentation)': 'aug_solarize', 'layer scale': 'pre-train_layer_scale',
                 'amp (for training)': 'amp', 'drop path rate': 'drop_path_rate', 'weight decay': 'weight_decay',
                 'grayscale (augmentation)': 'aug_grayscale', 'crop (augmentation)': 'aug_crop', 'min lr': 'min_lr',
                 'flip (augmentation)': 'aug_flip', 'prefetch factor': 'pre-train_prefetch_factor',
                 'warmup lr': 'pre-train_warmup_lr', 'optimizer eps': 'pre-train_opt_eps',

                 # ----------------------- METRICS --------------------
                 'inference VRAM @32 [GB]': 'inference_memory_@32', 'inference VRAM @128 [GB]': 'inference_memory_@128',
                 'inference VRAM @1 [GB]': 'inference_memory_@1', 'inference VRAM @64 [GB]': 'inference_memory_@64',

                 'total finetuning time [h*GPUs]': 'final_time_sum', 'total validation time [h*GPUs]': 'final_validation_time_sum',

                 'throughput [ims/s]': 'throughput_value', 'throughput batch size [ims]': 'throughput_batch_size',

                 'training VRAM [GB]': 'peak_memory_total', 'training VRAM (single GPU) [GB]': 'peak_memory_single',

                 'number of parameters [Millions]': 'number of parameters', 'GFLOPs': 'flops',

                 'validation loss': 'final_validation_loss', 'training loss': 'final_loss',

                 'top-5 validation accuracy': 'top_val_acc5', 'top-5 training accuracy': 'top_acc5',
                 'top-1 validation accuracy': 'top_val_acc1', 'top-1 training accuracy': 'top_acc1',
                 }


_PER_EPOCH_METRICS = {'gradient norm (max)': 'grad norm max', 'gradient norm (infinities)': 'inf grad norm',
                      'gradient norm (80-th percentile)': 'grad norm 80%', 'gradient norm (mean)': 'grad norm avrg',
                      'gradient norm (20-th percentile)': 'grad norm 20%',

                      'learning rate': 'learning rate',

                      'training time per epoch [h*GPUs]': 'time', 'validation time per epoch [h*GPUs]': 'validation_time',
                      'training time (total) [h*GPUs]': 'time_sum', 'validation time (total) [h*GPUs]': 'validation_time_sum',

                      'validation loss': 'validation_loss', 'training loss': 'loss',

                      'top-5 validation accuracy': 'val_acc5', 'top-5 training accuracy': 'acc5',
                      'top-1 training accuracy': 'acc1', 'top-1 validation accuracy': 'val_acc1',
                      }

_METRIC_CONVERSION_FACTOR = {
    'inference VRAM @1 [GB]': 1024**3, 'inference VRAM @32 [GB]': 1024**3, 'inference VRAM @64 [GB]': 1024**3,
    'inference VRAM @128 [GB]': 1024**3, 'total finetuning time [h*GPUs]': 60**2,
    'total validation time [h*GPUs]': 60**2, 'training VRAM [GB]': 1024**3, 'training VRAM (single GPU) [GB]': 1024**3,
    'number of parameters [Millions]': 1e6, 'GFLOPs': 1e9, 'training time per epoch [h*GPUs]': 60**2,
    'validation time per epoch [h*GPUs]': 60**2, 'training time (total) [h*GPUs]': 60**2,
    'validation time (total) [h*GPUs]': 60**2,
}


class _DummyFile():
    def write(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def no_print():
    save_stdout = sys.stdout
    sys.stdout = _DummyFile()
    yield
    sys.stdout = save_stdout


def load_data(file_name=None, order_by_date=False, include_run_name=False):
    if file_name is None:
        file_name = _DATA_FILE
    with open(file_name, 'r') as f:
        try:
            runs = json.load(f)
        except JSONDecodeError:
            sleep(10)
            runs = json.load(f)

    runs = [run for run in runs if 'model' in run and run['model'] is not None and 'run_date' in run and run['run_date'] is not None]

    run_data = {metr_name: [run[metr_id] if metr_id in run else None for run in runs]
                for metr_name, metr_id in _COLUMN_NAMES.items()}
    run_data = {k: [v_i / _METRIC_CONVERSION_FACTOR[k] if (isinstance(v_i, int) or isinstance(v_i, float)) and k in _METRIC_CONVERSION_FACTOR else v_i for v_i in v]
                for k, v in run_data.items()}
    with no_print():
        run_data['taxonomy class'] = [tx.get_taxonomy_class(name) for name in run_data['model']]
    run_data['model'] = [tx.get_model_name(name) for name in run_data['model']]
    run_data['epoch_data'] = [{ep: {k: ep_data[k_old] / (_METRIC_CONVERSION_FACTOR[k] if k in _METRIC_CONVERSION_FACTOR else 1.)
                                               for k, k_old in _PER_EPOCH_METRICS.items() if k_old in ep_data} for ep, ep_data in run['epoch_data'].items()} for run in runs]
    run_data['epoch_data'] = [json.dumps(run) for run in run_data['epoch_data']]
    df = pd.DataFrame(run_data)

    cols_first = ['run name', 'model', 'taxonomy class', 'top-1 validation accuracy', 'number of parameters [Millions]',
                  'GFLOPs',
                  'throughput [ims/s]', 'throughput batch size [ims]', 'training VRAM [GB]',
                  'training VRAM (single GPU) [GB]',
                  'inference VRAM @1 [GB]', 'inference VRAM @32 [GB]', 'inference VRAM @64 [GB]',
                  'inference VRAM @128 [GB]',
                  'total finetuning time [h*GPUs]', 'total validation time [h*GPUs]', 'validation loss', 'training loss',
                  'top-5 validation accuracy', 'top-1 training accuracy', 'top-5 training accuracy']

    if order_by_date:
        cols_first.insert(1, 'run date')

    rest_cols = set(_COLUMN_NAMES.keys()).difference(set(cols_first))
    finetuning_cols = {col for col in rest_cols if '(finetuning)' in col}
    rest_cols = rest_cols.difference(finetuning_cols)
    pretraining_cols = {col for col in rest_cols if '(pretrining)' in col}
    rest_cols = rest_cols.difference(pretraining_cols)
    augmentation_cols = {col for col in rest_cols if '(augmentation)' in col}
    rest_cols = rest_cols.difference(augmentation_cols)
    columns = cols_first + sorted(list(finetuning_cols)) + sorted(list(augmentation_cols)) + sorted(list(rest_cols)) \
              + sorted(list(pretraining_cols)) + ['epoch_data']

    if not include_run_name:
        columns.remove('run name')

    df['run date'] = pd.to_datetime(df['run date'], format=_DATETIME_FORMAT)
    if order_by_date:
        df = df.sort_values('run date', ascending=False)
    else:
        df = df.sort_values(['taxonomy class', 'model'])
    return df.to_dict('records'), columns


def prepare_table_info(file_name=None, order_by_date=False, include_run_name=False):
    data, columns = load_data(file_name=file_name, order_by_date=order_by_date, include_run_name=include_run_name)
    cols = [{'name': c, 'id': c} for c in columns if c != 'epoch_data']
    tooltips = {c: {'value': c, 'use_with': 'header'} for c in columns}

    return data, cols, tooltips
