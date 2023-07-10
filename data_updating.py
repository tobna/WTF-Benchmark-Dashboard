import json
import os
import re
from time import time, sleep
from multiprocessing import Pool, Process


def extract_run_data(logfile, max_infors_per_line=10):
    device_re = re.compile("INFO: training on (.*) -> (.*)\n")
    args_re = re.compile("INFO: full set of arguments: (.*)\n")
    old_args_re = re.compile("INFO: full set of old arguments: (.*)\n")
    exp_id_re = re.compile("INFO: .* experiment_id=(.*)\n")
    run_date_re = re.compile("INFO: Run name: .*_(\d+\.\d+\.\d+)_(\d+:\d+:\d+).*")
    run_name_re = re.compile("INFO: Run name: '(.*)'")
    epoch_data_re = re.compile(
        "INFO: epoch (\d+): (?:([^=,]*)=([^=,]*))" + (max_infors_per_line - 1) * "(?:, ([^=]*)=([^=,]*))?" + "\n")
    eff_metrics_re = re.compile("INFO: Efficiency metrics: (?:([^=,]*)=([^=,]*))" + (
            max_infors_per_line - 1) * "(?:, ([^=]*)=([^=,]*))?" + "\n")
    eff_metrics_json_re = re.compile("INFO: (?:Efficiency m|M)etrics: (.*)\n")

    lines = []
    with open(logfile, 'r') as f:
        for line in f:
            lines.append(line)

    run_data = {"epoch_data": {}}
    for line in lines:
        device = device_re.search(line)
        if device is not None:
            # print(f"\tfound device {device.group(2)}")
            run_data['device'] = device.group(2)
            continue

        args = args_re.search(line)
        if args is not None:
            args = args.group(1).replace("'", '"').replace("None", "null").replace(" False", " false").replace(" True",
                                                                                                               " true")
            try:
                args = json.loads(args)
                run_data = run_data | args
            except json.decoder.JSONDecodeError as err:
                print(f"JSONDecodeError {err}\n\t when trying to decode {args}\n\t of file {logfile}")
            # print(f"\tfound args {args}")
            continue

        old_args = old_args_re.search(line)
        if old_args is not None:
            old_args = old_args.group(1).replace("'", '"').replace("None", "null").replace(" False", " false")\
                .replace(" True", " true")
            old_args = json.loads(old_args)

            if 'task' not in old_args:
                continue
            old_task = old_args['task']
            if 'task' in run_data and 'eval' in run_data['task']:
                run_data = old_args | run_data
            else:
                run_data = {f"{old_task}_{key}": val for key, val in old_args.items()} | run_data
            continue

        exp_id = exp_id_re.search(line)
        if exp_id is not None:
            # print(f"\tfound exp id {exp_id.group(1)}")
            run_data['experiment_id'] = int(exp_id.group(1))
            continue

        run_date = run_date_re.search(line)
        if run_date is not None:
            # print(f"\trun was at {run_date.group(1)} {run_date.group(2)}")
            run_data['run_date'] = f"{run_date.group(1)} {run_date.group(2)}"
            if 'run_name' not in run_data:
                run_name = run_name_re.search(line)
                assert run_name is not None, f"Run name not found in line {line}"
                run_data['run_name'] = run_name.group(1)
            continue

        epoch_data = epoch_data_re.search(line)
        if epoch_data is not None:
            # print(f"\tfound epoch data {epoch_data.groups()}")
            epoch = int(epoch_data.group(1))
            if epoch not in run_data["epoch_data"]:
                run_data["epoch_data"][epoch] = {}
            data_list = list(epoch_data.groups()[1:])
            while len(data_list) >= 2 and data_list[0] is not None:
                key = data_list[0]
                val = data_list[1]

                try:
                    val = float(val)
                except ValueError:
                    pass

                if isinstance(val, str):
                    if val.endswith("s"):
                        val = float(val[:-1])
                    elif val.endswith("%"):
                        val = float(val[:-1]) / 100
                    elif val.startswith("[") and val.endswith("]"):
                        val = json.loads(val)
                        if len(val) == 1:
                            val = val[0]

                run_data["epoch_data"][epoch][key] = val
                # print(f"adding {key}: {val} ({type(val)})", end='\t')
                data_list = data_list[2:]
            # print("")
            continue

        efficiency_data = eff_metrics_re.search(line)
        if efficiency_data is not None:
            data_list = list(efficiency_data.groups())
            while len(data_list) >= 2 and data_list[0] is not None:
                key = data_list[0]
                val = data_list[1]

                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass

                run_data[key] = val
                data_list = data_list[2:]
            continue

        efficiency_data = eff_metrics_json_re.search(line)
        if efficiency_data is not None:
            efficiency_data = json.loads(efficiency_data.group(1).replace("'", '"'))
            # has_through_b4 = 'throughput' in run_data
            # if has_through_b4:
            #     print(f"throughput twice in file {logfile}")
            #     print(f"old vals: {run_data['throughput']}")
            run_data = run_data | efficiency_data
            # if has_through_b4:
            #     print(f"new vals: {efficiency_data} -> into dict: {run_data['throughput']}")
            continue

    # make times to be GPU seconds
    num_gpus = run_data['world_size'] if 'world_size' in run_data else -1
    for epoch in run_data['epoch_data'].values():
        for key in epoch.keys():
            if 'time' in key:
                epoch[key] = num_gpus * epoch[key]

    # calculate time sums and correct some typos...
    train_time_sum = val_time_sum = 0.
    epochs = sorted(run_data['epoch_data'].keys())
    for ep_n in epochs:
        ep = run_data['epoch_data'][ep_n]
        if 'time' in ep:
            train_time_sum += ep['time']
            ep['time_sum'] = train_time_sum
        if 'validation_time' in ep:
            val_time_sum += ep['validation_time']
            ep['validation_time_sum'] = val_time_sum
        if 'validataion_accuracy' in ep:
            ep['val_acc1'] = ep['validataion_accuracy']
            ep.pop('validataion_accuracy')

    # if accuracy is accidentally *10_000 instead of *100, correct that shit
    acc_list = [0.] + [ep['acc1'] for ep in run_data['epoch_data'].values() if 'acc1' in ep]
    max_acc = max(acc_list)
    if max_acc > 1.:
        # accuracy is too high by a factor of 100
        for ep in run_data['epoch_data'].values():
            if 'acc1' in ep:
                ep['acc1'] = ep['acc1'] / 100

    acc_val_list = [0.] + [ep['val_acc1'] for ep in run_data['epoch_data'].values() if 'val_acc1' in ep]
    max_acc_val = max(acc_val_list)
    if max_acc_val > 1.:
        for ep in run_data['epoch_data'].values():
            if 'val_acc1' in ep:
                ep['val_acc1'] = ep['val_acc1'] / 100
    # run_data['epoch_data'] = json.dumps(run_data['epoch_data'])

    # final epoch data extraction
    if len(run_data['epoch_data']) > 0:
        final_epoch = run_data['epoch_data'][max(run_data['epoch_data'].keys())]
        run_data = run_data | {f"final_{key}": val for key, val in final_epoch.items()}
        run_data['final_epoch'] = max(run_data['epoch_data'].keys())

    # collapse throughput dict
    if 'throughput' in run_data:
        for key, val in run_data['throughput'].items():
            run_data[f'throughput_{key}'] = val
        run_data.pop('throughput')

    # best epoch data extraction
    if len(run_data['epoch_data']) > 0:
        keys = set(run_data['epoch_data'][min(run_data['epoch_data'].keys())]).union(set(run_data['epoch_data'][max(run_data['epoch_data'].keys())]))
        for key in keys:
            if 'acc' in key:
                run_data['top_'+key] = max([0.] + [run[key] for run in run_data['epoch_data'].values() if key in run])

    if 'world_size' in run_data and 'batch_size' in run_data:
        run_data['local_batch_size'] = run_data['batch_size']
        run_data['batch_size'] *= run_data['world_size']

    if 'pre-train_world_size' in run_data and 'pre-train_batch_size' in run_data:
        run_data['pre-train_local_batch_size'] = run_data['pre-train_batch_size']
        run_data['pre-train_batch_size'] *= run_data['pre-train_world_size']

    return run_data


log_folder = "/netscratch/nauen/EfficientCVBench/logging/"
data_file_name = "data_tmp.json"


def _data_process(n_workers, update_interval):
    while True:
        start = time()
        logfiles = [log_folder + f.split('/')[-1] for f in os.listdir(log_folder) if f.endswith('.log')]
        with Pool(n_workers) as p:
            runs = p.map(extract_run_data, logfiles)
        runs = [run for run in runs if 'run_name' in run and run['run_name'] is not None and len(run['run_name']) > 0]
        with open('data.tmp', "w+") as f:
            json.dump(runs, f)
        os.replace('data.tmp', data_file_name)
        sleep_time = max(update_interval - time() + start, 0)
        sleep(sleep_time)


def start_data_process(n_workers=5, update_interval=10):
    data_process = Process(target=_data_process, args=(n_workers, update_interval, ))
    data_process.start()
    return data_process
