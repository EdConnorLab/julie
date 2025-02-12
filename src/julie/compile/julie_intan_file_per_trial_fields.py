from clat.compile.task.task_field import TaskField
from clat.intan import spike_file
from clat.intan.channels import Channel
from clat.intan.livenotes import map_task_id_to_epochs_with_livenotes
from clat.intan.marker_channels import epoch_using_marker_channels
import os
import re


class SpikeTimesForChannelsField(TaskField):
    def __init__(self, intan_data_path: str, name: str = "SpikeTimes"):
        super().__init__(name)
        self.intan_data_path = intan_data_path

    def get(self, task_id: int) -> dict[Channel, list[float]] | None:
        matching_intan_file_paths = find_matching_directories(self.intan_data_path, task_id)
        if len(matching_intan_file_paths) == 0:
            return None
        intan_file_path = matching_intan_file_paths[-1]
        spike_path = os.path.join(intan_file_path, "spike.dat")
        digital_in_path = os.path.join(intan_file_path, "digitalin.dat")
        notes_path = os.path.join(intan_file_path, "notes.txt")

        spike_tstamps_for_channels, sample_rate = spike_file.fetch_spike_tstamps_from_file(spike_path)

        stim_epochs_from_markers = epoch_using_marker_channels(digital_in_path)
        epochs_for_task_ids = map_task_id_to_epochs_with_livenotes(notes_path,
                                                                   stim_epochs_from_markers)
        spikes_for_channels = filter_spikes_with_epochs(spike_tstamps_for_channels,
                                                        epochs_for_task_ids, task_id,
                                                        sample_rate=sample_rate)
        return spikes_for_channels


def filter_spikes_with_epochs(spike_tstamps_for_channels: dict[Channel, list[float]],
                              epochs_for_task_ids: dict[int, tuple[int, int]], task_id: int,
                              sample_rate: float = 30000) -> dict[
    Channel, list[float]]:
    filtered_spikes_for_channels = {}
    epoch = epochs_for_task_ids[task_id]
    for channel, tstamps in spike_tstamps_for_channels.items():
        passed_filter = []
        for spike_tstamp in tstamps:
            spike_index = int(spike_tstamp * sample_rate)
            if epoch[0] <= spike_index <= epoch[1]:
                passed_filter.append(spike_tstamp)
        filtered_spikes_for_channels[channel] = passed_filter
    return filtered_spikes_for_channels


class EpochStartStopField(TaskField):
    # TODO: clean up a bit, we're duplicating the epoch retrieval between this and SpikeTimesForChannelsField
    def __init__(self, intan_data_path: str, name: str = "EpochStartStop"):
        super().__init__(name)
        self.intan_data_path = intan_data_path

    def get(self, task_id: int) -> tuple[float, float] | None:
        matching_intan_file_paths = find_matching_directories(self.intan_data_path, task_id)
        if len(matching_intan_file_paths) == 0:
            return None
        intan_file_path = matching_intan_file_paths[-1]
        spike_path = os.path.join(intan_file_path, "spike.dat")
        digital_in_path = os.path.join(intan_file_path, "digitalin.dat")
        notes_path = os.path.join(intan_file_path, "notes.txt")

        if len(matching_intan_file_paths) == 0:
            return None

        _, sample_rate = spike_file.fetch_spike_tstamps_from_file(spike_path)
        stim_epochs_from_markers = epoch_using_marker_channels(digital_in_path)
        epochs_for_task_ids = map_task_id_to_epochs_with_livenotes(notes_path,
                                                                   stim_epochs_from_markers)
        epoch = epochs_for_task_ids[task_id]
        epoch_start = epoch[0] / sample_rate
        epoch_stop = epoch[1] / sample_rate
        return epoch_start, epoch_stop


def find_matching_directories(root_folder: str, target_number: int) -> list:
    """
    Search through a folder to find directories that start with the given target_number,
    sorted based on the text that comes after the target_number and underscore.

    Parameters:
        root_folder (str): The path of the folder to search in.
        target_number (int): The target number to search for.

    Returns:
        list: A list of full directory paths that match the target_number, sorted by the text after the number.
    """
    matching_dirs = []
    for dirname in os.listdir(root_folder):
        if re.match(f'^{str(target_number)}_', dirname):
            full_path = os.path.join(root_folder, dirname)
            matching_dirs.append(full_path)

    # Sort the list based on the portion of the directory name that comes after the target_number and underscore
    matching_dirs.sort(key=lambda x: x.split(f'{target_number}_')[-1])

    return matching_dirs
