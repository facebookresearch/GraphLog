"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""

import os
from codes.logbook import filesystem_logger as fs_log
from codes.utils.util import flatten_dict, get_main_working_dir


class LogBook:
    def __init__(self, config):
        self._experiment_id = 0
        self.metrics_to_omit = ["mode"]

        flattened_config = flatten_dict(config.to_serializable_dict(), sep="_")
        self.should_use_tb = False
        fs_log.set_logger(config=config)

    def _log_metrics(self, dic, prefix, step):
        """Method to log metric"""
        formatted_dict = {}
        for key, val in dic.items():
            formatted_dict[prefix + "_" + key] = val

    def write_config_log(self, config):
        """Write config"""
        fs_log.write_config_log(config)
        flatten_config = flatten_dict(config, sep="_")
        flatten_config["experiment_id"] = self._experiment_id

    def write_metric_logs(self, metrics):
        """Write Metric"""
        metrics["experiment_id"] = self._experiment_id
        fs_log.write_metric_logs(metrics)
        flattened_metrics = flatten_dict(metrics, sep="_")

        if self.metrics_to_omit:
            metric_dict = {
                key: flattened_metrics[key]
                for key in flattened_metrics
                if key not in self.metrics_to_omit
            }
        else:
            metric_dict = flattened_metrics
        prefix = metrics.get("mode", None)
        num_timesteps = metric_dict.pop("minibatch")
        # print(metric_dict)
        self._log_metrics(dic=metric_dict, prefix=prefix, step=num_timesteps)

        if self.should_use_tb:

            timestep_key = "num_timesteps"
            for key in set(list(metrics.keys())) - set([timestep_key]):
                self.tensorboard_writer.add_scalar(
                    tag=key,
                    scalar_value=metrics[key],
                    global_step=metrics[timestep_key],
                )

    def write_compute_logs(self, **kwargs):
        """Write Compute Logs"""
        kwargs["experiment_id"] = self._experiment_id
        fs_log.write_metric_logs(**kwargs)
        metric_dict = flatten_dict(kwargs, sep="_")

        num_timesteps = metric_dict.pop("num_timesteps")
        self._log_metrics(dic=metric_dict, step=num_timesteps, prefix="compute")

    def write_message_logs(self, message):
        """Write message logs"""
        fs_log.write_message_logs(message, experiment_id=self._experiment_id)

    def write_metadata_logs(self, metadata):
        """Write metadata"""
        metadata["experiment_id"] = self._experiment_id
        fs_log.write_metadata_logs(metadata)
        # self.log_other(key="best_epoch_index", value=kwargs["best_epoch_index"])
