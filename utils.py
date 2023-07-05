import json
import os
import pandas as pd
import taxonomy as tx

_DATA_FILE = os.path.join('data', 'data.json')
_DATETIME_FORMAT = '%d.%m.%Y %H:%M:%S'
# ----------------------- INFO --------------------
_COLUMN_NAMES = {'run name': 'run_name', 'run date': 'run_date', 'model': 'model', 'GPU type (finetuning)': 'device',

                 # ----------------------- HYPER PARAMS --------------------
                 'epochs (pretraining)': 'pre-train_epochs', 'image resolution (pretraining)': 'pre-train_imsize',
                 'GPUS (pretraining)': 'pre-train_world_size', 'lr (pretraining)': 'pre-train_lr',
                 'per-GPU batch size (pretraining)': 'pre-train_local_batch_size',
                 'dataloader workers (pretraining)': 'pre-train_num_workers',
                 'batch size (pretraining)': 'pre-train_batch_size',
                 'dataset (pretraining)': 'pre-train_dataset',

                 'image resolution (finetuning)': 'imsize', 'epochs (finetuning)': 'final_epoch',
                 'GPUs (finetuning)': 'world_size', 'per-GPU batch size (finetuning)': 'local_batch_size',
                 'lr (finetuning)': 'lr', 'dataloader workers (finetuning)': 'num_workers',
                 'dataset (finetuning)': 'dataset', 'batch size (finetuning)': 'batch_size',

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
                 'inference VRAM @32': 'inference_memory_@32', 'inference VRAM @128': 'inference_memory_@128',
                 'inference VRAM @1': 'inference_memory_@1', 'inference VRAM @64': 'inference_memory_@64',

                 'total finetuning time': 'final_time_sum', 'total validation time': 'final_validation_time_sum',

                 'throughput': 'throughput_value', 'throughput batch size': 'throughput_batch_size',

                 'training VRAM': 'peak_memory_total', 'training VRAM (single GPU)': 'peak_memory_single',

                 'number of parameters': 'number of parameters', 'FLOPs': 'flops',

                 'validation loss': 'final_validation_loss', 'training loss': 'final_loss',

                 'top-5 validation accuracy': 'top_val_acc5', 'top-5 training accuracy': 'top_acc5',
                 'top-1 validation accuracy': 'top_val_acc1', 'top-1 training accuracy': 'top_acc1',
                 }


_PER_EPOCH_METRICS = {'gradient norm (max)': 'grad norm max', 'gradient norm (infinities)': 'inf grad norm',
                      'gradient norm (80-th percentile)': 'grad norm 80%', 'gradient norm (mean)': 'grad norm avrg',
                      'gradient norm (20-th percentile)': 'grad norm 20%',

                      'learning rate': 'learning rate',

                      'training time per epoch': 'time', 'validation time per epoch': 'validation_time',
                      'training time (total)': 'time_sum', 'validation time (total)': 'validation_time_sum',

                      'validation loss': 'validation_loss', 'training loss': 'loss',

                      'top-5 validation accuracy': 'val_acc5', 'top-5 training accuracy': 'acc5',
                      'top-1 training accuracy': 'acc1', 'top-1 validation accuracy': 'val_acc1',
                      }


def prepare_table_info():
    with open(_DATA_FILE, 'r') as f:
        runs = json.load(f)

    run_data = {metr_name: [run[metr_id] if metr_id in run else None for run in runs]
                for metr_name, metr_id in _COLUMN_NAMES.items()}
    run_data['taxonomy class'] = [tx.get_taxonomy_class(name) for name in run_data['model']]
    run_data['model'] = [tx.get_model_name(name) for name in run_data['model']]
    run_data['epoch_data'] = [json.dumps({ep: {k: ep_data[k_old] for k, k_old in _PER_EPOCH_METRICS.items() if k_old in ep_data} for ep, ep_data in run['epoch_data'].items()}) for run in runs]
    df = pd.DataFrame(run_data)

    cols_first = ['run name', 'model', 'taxonomy class', 'top-1 validation accuracy', 'number of parameters', 'FLOPs',
                  'throughput', 'throughput batch size', 'training VRAM', 'training VRAM (single GPU)',
                  'inference VRAM @1', 'inference VRAM @32', 'inference VRAM @64', 'inference VRAM @128',
                  'total finetuning time', 'total validation time', 'validation loss', 'training loss',
                  'top-5 validation accuracy', 'top-1 training accuracy', 'top-5 training accuracy']

    rest_cols = set(_COLUMN_NAMES.keys()).difference(set(cols_first))
    finetuning_cols = {col for col in rest_cols if '(finetuning)' in col}
    rest_cols = rest_cols.difference(finetuning_cols)
    pretraining_cols = {col for col in rest_cols if '(pretrining)' in col}
    rest_cols = rest_cols.difference(pretraining_cols)
    augmentation_cols = {col for col in rest_cols if '(augmentation)' in col}
    rest_cols = rest_cols.difference(augmentation_cols)
    columns = cols_first + sorted(list(finetuning_cols)) + sorted(list(augmentation_cols)) + sorted(list(rest_cols)) \
              + sorted(list(pretraining_cols)) + ['epoch_data']

    df['run date'] = pd.to_datetime(df['run date'], format=_DATETIME_FORMAT)
    df = df.sort_values('model')
    df = df.sort_values('taxonomy class')

    cols = [{'name': c, 'id': c} for c in columns if c != 'epoch_data']
    tooltips = {c: {'value': c, 'use_with': 'header'} for c in columns}

    return df.to_dict('records'), cols, tooltips
