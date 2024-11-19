import territorytools.behavior as tdt
import json
import os
import yaml


def make_dict(info_string, delimiter):
    """
    Function to make dicts from strings separated by a given delimiter. Input string will be separated by delimiter and
    every odd element will be used as a key in the output dict, with each associated value being the even elements.
    e.g. 'Name_Dave_Year_3' with delimiter='_'  ->  {'Name': 'Dave', 'Year': '3'} (note String conversion of 3)

    Parameters
    ----------
    info_string : str
        string to parse into a Dict
    delimiter : str
        string used to separate the metadata in info_string

    Returns
    -------
    dict
        dictionary with the given info/metadata
    """
    file_parts = info_string.split(delimiter)
    out_dict = {}
    c = 0
    for k in file_parts:
        if c % 2 == 0:
            out_dict[k] = file_parts[c+1]
        c += 1
    return out_dict


def get_key_val_recur(key_name: str, nest_dict):
    key_split = key_name.split('/')
    if len(key_split) > 1:
        n_dict = nest_dict[key_split[0]]
        out_val = get_key_val_recur(''.join(key_split[1:]), nest_dict=n_dict)
    else:
        out_val = nest_dict[key_split[0]]
    return out_val


def set_key_val_recur(key_name: str, value, nest_dict):
    key_split = key_name.split('/')
    if len(key_split) > 1:
        n_dict = nest_dict[key_split[0]]
        set_key_val_recur(''.join(key_split[1:]), value, nest_dict=n_dict)
    elif key_name in nest_dict.keys():
        nest_dict[key_name] = value


class BasicRun:
    """
    A class to represent a basic recording for an experiment. Most applications will extend this class for a specific
    kind of recording (ephys, imaging, behavior, etc.)

    Attributes
    ----------
    data : dict
        dataset recorded in this run. keys and values are arbitrary and depend on the experiment
    info_dict : dict
        metadata associated with the run. used by BasicExp to filter and group recordings based on given keys/values

    Methods
    -------
    get_key_val(key_name):
        gets value of given key in this run's metadata
    add_key_val(keyname

    """
    def __init__(self, data, info_dict):
        self.info = info_dict
        self.data = data

    def get_key_val(self, key_name: str, nest_dict=None):
        if nest_dict is None:
            nest_dict = self.info
        return get_key_val_recur(key_name, nest_dict)

    def add_key_val(self, key_name, key_val):
        self.info[key_name] = key_val


class RunFromFile(BasicRun):
    def __init__(self, file_name, info_dict, rot_offset=0, num_frames=0):
        super().__init__(tdt.get_territory_data(file_name, rot_offset, num_frames), info_dict)


class ComputeRun(BasicRun):
    def __init__(self, run_obj, func, *args, with_info=False):
        if with_info:
            new_data = func(run_obj.data, run_obj.info, *args)
        else:
            new_data = func(run_obj.data, *args)
        super().__init__(new_data, run_obj.info)


class BasicExp:
    def __init__(self, runs):
        self.runs = runs

    def compute_across_runs(self, func, *args, pass_info=False):
        run_list = []
        for r in self.runs:
            run_list.append(ComputeRun(r, func, *args, with_info=pass_info))
        return BasicExp(run_list)

    def compute_across_groups(self, key_name, func, *args, arg_per_group=False):
        group_out = dict()
        group_ids, _ = self.unique_key_vals(key_name)
        for ind, g in enumerate(group_ids):
            group_runs = self.get_runs_from_key_val(key_name, g)
            group_data, group_info = self.get_group_data_info(group_runs)
            if arg_per_group:
                group_out[g] = func(g, group_data, group_info, args[0][ind])
            else:
                group_out[g] = func(g, group_data, group_info, *args)
        return group_out

    def compute_across_group(self, key_name, key_val, func, *args):
        group_runs = self.get_runs_from_key_val(key_name, key_val)
        group_data, group_info = self.get_group_data_info(group_runs)
        group_out = func(key_val, group_data, group_info, *args)
        return group_out

    def unique_key_vals(self, key_name):
        unique_key_vals = []
        for r in self.runs:
            val = r.get_key_val(key_name)
            if val not in unique_key_vals:
                unique_key_vals.append(val)
        return unique_key_vals, len(unique_key_vals)

    def get_runs_from_key_val(self, key_name, key_val):
        flagged_runs = []
        for r in self.runs:
            if r.get_key_val(key_name) == key_val:
                flagged_runs.append(r)
        return flagged_runs

    def get_group_data_info(self, group_runs):
        group_data = []
        group_info = []
        for r in group_runs:
            group_data.append(r.data)
            group_info.append(r.info)
        return group_data, group_info

    def filter_by_group(self, key_name, key_val):
        return BasicExp(self.get_runs_from_key_val(key_name, key_val))

    def get_run_data(self):
        run_data = [r.data for r in self.runs]
        return run_data


class MDcontroller:
    def __init__(self, *args):
        if len(args) > 0:
            self.file_name = args[0]
        self.metadata = load_metadata(self.file_name)

    def add_metadata(self, key: str, value: str or dict):
        """
        Add key/value pair to Controllers metadata in place, OVERWRITES EXISTING KEYS

        Parameters
        ----------
        key: key to add to metadata
        value: value for that key

        """
        self.metadata[key] = value

    def remove_metadata(self, key: str):
        """
        Remove given key from this Controller's metadata in place

        Parameters
        ----------
        key: key to be removed

        """
        self.metadata.pop(key)

    def has_kv(self, key: str, value: str):
        return has_key_value_recur(self.metadata, key, value)

    def get_val(self, key):
        return get_key_val_recur(key, self.metadata)

    def set_key_val(self, key_name: str, value):
        set_key_val_recur(key_name, value, self.metadata)

    def __str__(self):
        return dict_to_yaml(self.metadata)

    def save_metadata(self, save_path):
        yaml_file = open(save_path, 'w')
        yaml.safe_dump(self.metadata, yaml_file)


def dict_to_yaml(md_dict: dict, indent=0):
    out_str = ''
    for k in md_dict.keys():
        for i in range(indent):
            out_str += '\t'
        out_str += k
        if type(md_dict[k]) == dict:
            out_str += '\n'
            out_str += dict_to_yaml(md_dict[k], indent=indent + 1)
        else:
            out_str += ': ' + str(md_dict[k]) + '\n'
    no_quote = out_str.replace('\'', '')
    return no_quote


def has_key_value_recur(test_dict: dict, key: str, value: str):
    """
    Recursively check if given key/value pair exists in a test dictionary

    Parameters
    ----------
    test_dict: dict to recursively search through
    key: key to query
    value: value to query for the given key

    Returns
    -------
    has_kv: boolean whether the given key value was found
    """
    has_kv = False
    for k in test_dict.keys():
        if has_kv:
            return has_kv
        if type(test_dict[k]) == dict:
            has_kv = has_key_value_recur(test_dict[k], key, value)
        elif test_dict[k] == value:
            has_kv = True
    return has_kv


def load_metadata(metadata_file: str):
    """
    Wrapper for loading metadata via json and yaml libraries

    Parameters
    ----------
    metadata_file: path to metadata file (must be .yaml or .json)

    Returns
    -------
    metadata_dict: dictionary containing metadata as key/value pairs

    """

    file_ext = metadata_file.split('.')[-1]
    file_handle = open(metadata_file, 'r')
    metadata_dict = dict()
    if file_ext == 'json':
        metadata_dict = json.load(file_handle)
    elif file_ext == 'yaml' or file_ext == 'yml':
        metadata_dict = yaml.safe_load(file_handle)
    else:
        raise Exception("Metadata file must be either .json or .yaml")
    return metadata_dict


def yaml_to_json(yaml_file: str):
    """
    Converts yaml files to json files

    Parameters
    ----------
    yaml_file: path to yaml file which will be converted

    Returns
    -------
    json_file: path to newly created json file

    """
    metadata_dict = load_metadata(yaml_file)
    yaml_pref = yaml_file.split('.')[:-1]
    json_path = yaml_pref + '.json'
    new_json = open(json_path, 'w')
    json.dump(metadata_dict, new_json)
    return json_path


def sniff_metadata(root_dir, target_key, target_value):
    """
    Returns paths to all locations of metadata files which contain the given target metadata

    Parameters
    ----------
    root_dir: root folder to recursively search through
    target_key: metadata key to look for
    target_value: metadata value to look for

    Returns
    -------
    path_list: list of all paths to metadata files which contain the requested metadata
    """
    path_list = []
    for f in os.listdir(root_dir):
        this_path = root_dir + '/' + f
        if os.path.isdir(this_path):
            path_list += sniff_metadata(this_path, target_key, target_value)
        else:
            file_ext = f.split('.')[-1]
            if file_ext == 'yaml' or file_ext == 'yml':
                mdc = MDcontroller(this_path)
                if mdc.has_kv(target_key, target_value):
                    path_list.append(this_path)
    return path_list
