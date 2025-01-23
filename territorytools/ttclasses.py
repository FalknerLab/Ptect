import territorytools.behavior as tdt
import json
import os
import yaml


def make_dict(info_string, delimiter):
    """
    Creates a dictionary from a string separated by a given delimiter.

    Parameters
    ----------
    info_string : str
        String to parse into a dictionary.
    delimiter : str
        String used to separate the metadata in `info_string`.

    Returns
    -------
    dict
        Dictionary with the given info/metadata.
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
    """
    Recursively retrieves the value of a nested key in a dictionary.

    Parameters
    ----------
    key_name : str
        Key name, with nested keys separated by '/'.
    nest_dict : dict
        Dictionary to search.

    Returns
    -------
    any
        Value associated with the key.
    """
    key_split = key_name.split('/')
    if len(key_split) > 1:
        n_dict = nest_dict[key_split[0]]
        out_val = get_key_val_recur(''.join(key_split[1:]), nest_dict=n_dict)
    else:
        out_val = nest_dict[key_split[0]]
    return out_val


def set_key_val_recur(key_name: str, value, nest_dict):
    """
    Recursively sets the value of a nested key in a dictionary.

    Parameters
    ----------
    key_name : str
        Key name, with nested keys separated by '/'.
    value : any
        Value to set.
    nest_dict : dict
        Dictionary to update.

    Returns
    -------
    None
    """
    key_split = key_name.split('/')
    if len(key_split) > 1:
        n_dict = nest_dict[key_split[0]]
        set_key_val_recur(''.join(key_split[1:]), value, nest_dict=n_dict)
    elif key_name in nest_dict.keys():
        nest_dict[key_name] = value


class BasicRun:
    """
    Represents a basic recording for an experiment.

    Attributes
    ----------
    data : dict
        Dataset recorded in this run.
    info_dict : dict
        Metadata associated with the run.

    Methods
    -------
    get_key_val(key_name)
        Gets the value of a given key in this run's metadata.
    add_key_val(key_name, key_val)
        Adds a key-value pair to this run's metadata.
    """
    def __init__(self, data, info_dict):
        """
        Initializes a BasicRun instance.

        Parameters
        ----------
        data : dict
            Dataset recorded in this run.
        info_dict : dict
            Metadata associated with the run.
        """
        self.info = info_dict
        self.data = data

    def get_key_val(self, key_name: str, nest_dict=None):
        """
        Gets the value of a given key in this run's metadata.

        Parameters
        ----------
        key_name : str
            Key name to retrieve.
        nest_dict : dict, optional
            Nested dictionary to search (default is None).

        Returns
        -------
        any
            Value associated with the key.
        """
        if nest_dict is None:
            nest_dict = self.info
        return get_key_val_recur(key_name, nest_dict)

    def add_key_val(self, key_name, key_val):
        """
        Adds a key-value pair to this run's metadata.

        Parameters
        ----------
        key_name : str
            Key name to add.
        key_val : any
            Value to associate with the key.

        Returns
        -------
        None
        """
        self.info[key_name] = key_val


class RunFromFile(BasicRun):
    """
    Represents a run created from a file.

    Methods
    -------
    __init__(file_name, info_dict, rot_offset=0, num_frames=0)
        Initializes a RunFromFile instance.
    """
    def __init__(self, file_name, info_dict, rot_offset=0, num_frames=0):
        """
        Initializes a RunFromFile instance.

        Parameters
        ----------
        file_name : str
            Path to the file.
        info_dict : dict
            Metadata associated with the run.
        rot_offset : int, optional
            Rotation offset (default is 0).
        num_frames : int, optional
            Number of frames (default is 0).
        """
        super().__init__(tdt.get_territory_data(file_name, rot_offset, num_frames), info_dict)


class ComputeRun(BasicRun):
    """
    Represents a computed run based on another run.

    Methods
    -------
    __init__(run_obj, func, *args, with_info=False)
        Initializes a ComputeRun instance.
    """
    def __init__(self, run_obj, func, *args, with_info=False):
        """
        Initializes a ComputeRun instance.

        Parameters
        ----------
        run_obj : BasicRun
            The original run object.
        func : callable
            Function to compute new data.
        with_info : bool, optional
            Whether to pass info_dict to the function (default is False).
        """
        if with_info:
            new_data = func(run_obj.data, run_obj.info, *args)
        else:
            new_data = func(run_obj.data, *args)
        super().__init__(new_data, run_obj.info)


class BasicExp:
    """
    Represents an experiment consisting of multiple runs.

    Methods
    -------
    compute_across_runs(func, *args, pass_info=False)
        Computes a function across all runs.
    compute_across_groups(key_name, func, *args, arg_per_group=False)
        Computes a function across groups of runs.
    compute_across_group(key_name, key_val, func, *args)
        Computes a function across a specific group of runs.
    unique_key_vals(key_name)
        Gets unique values for a given key across all runs.
    get_runs_from_key_val(key_name, key_val)
        Gets runs that match a specific key-value pair.
    get_group_data_info(group_runs)
        Gets data and info for a group of runs.
    filter_by_group(key_name, key_val)
        Filters runs by a specific key-value pair.
    get_run_data()
        Gets data for all runs.
    """
    def __init__(self, runs):
        """
        Initializes a BasicExp instance.

        Parameters
        ----------
        runs : list of BasicRun
            List of runs in the experiment.
        """
        self.runs = runs

    def compute_across_runs(self, func, *args, pass_info=False):
        """
        Computes a function across all runs.

        Parameters
        ----------
        func : callable
            Function to compute.
        pass_info : bool, optional
            Whether to pass info_dict to the function (default is False).

        Returns
        -------
        BasicExp
            New experiment with computed runs.
        """
        run_list = []
        for r in self.runs:
            run_list.append(ComputeRun(r, func, *args, with_info=pass_info))
        return BasicExp(run_list)

    def compute_across_groups(self, key_name, func, *args, arg_per_group=False):
        """
        Computes a function across groups of runs.

        Parameters
        ----------
        key_name : str
            Key name to group by.
        func : callable
            Function to compute.
        arg_per_group : bool, optional
            Whether to pass different arguments for each group (default is False).

        Returns
        -------
        dict
            Dictionary with group keys and computed values.
        """
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
        """
        Computes a function across a specific group of runs.

        Parameters
        ----------
        key_name : str
            Key name to group by.
        key_val : any
            Key value to match.
        func : callable
            Function to compute.

        Returns
        -------
        any
            Computed value for the group.
        """
        group_runs = self.get_runs_from_key_val(key_name, key_val)
        group_data, group_info = self.get_group_data_info(group_runs)
        group_out = func(key_val, group_data, group_info, *args)
        return group_out

    def unique_key_vals(self, key_name):
        """
        Gets unique values for a given key across all runs.

        Parameters
        ----------
        key_name : str
            Key name to search.

        Returns
        -------
        list
            List of unique values.
        int
            Number of unique values.
        """
        unique_key_vals = []
        for r in self.runs:
            val = r.get_key_val(key_name)
            if val not in unique_key_vals:
                unique_key_vals.append(val)
        return unique_key_vals, len(unique_key_vals)

    def get_runs_from_key_val(self, key_name, key_val):
        """
        Gets runs that match a specific key-value pair.

        Parameters
        ----------
        key_name : str
            Key name to search.
        key_val : any
            Key value to match.

        Returns
        -------
        list of BasicRun
            List of runs that match the key-value pair.
        """
        flagged_runs = []
        for r in self.runs:
            if r.get_key_val(key_name) == key_val:
                flagged_runs.append(r)
        return flagged_runs

    def get_group_data_info(self, group_runs):
        """
        Gets data and info for a group of runs.

        Parameters
        ----------
        group_runs : list of BasicRun
            List of runs in the group.

        Returns
        -------
        list
            List of data for the group.
        list
            List of info for the group.
        """
        group_data = []
        group_info = []
        for r in group_runs:
            group_data.append(r.data)
            group_info.append(r.info)
        return group_data, group_info

    def filter_by_group(self, key_name, key_val):
        """
        Filters runs by a specific key-value pair.

        Parameters
        ----------
        key_name : str
            Key name to search.
        key_val : any
            Key value to match.

        Returns
        -------
        BasicExp
            New experiment with filtered runs.
        """
        return BasicExp(self.get_runs_from_key_val(key_name, key_val))

    def get_run_data(self):
        """
        Gets data for all runs.

        Returns
        -------
        list
            List of data for all runs.
        """
        run_data = [r.data for r in self.runs]
        return run_data


class MDcontroller:
    """
    Controller for managing metadata.

    Methods
    -------
    add_metadata(key, value)
        Adds a key-value pair to the metadata.
    remove_metadata(key)
        Removes a key from the metadata.
    has_kv(key, value)
        Checks if a key-value pair exists in the metadata.
    get_val(key)
        Gets the value of a key in the metadata.
    set_key_val(key_name, value)
        Sets the value of a key in the metadata.
    save_metadata(save_path)
        Saves the metadata to a file.
    """
    def __init__(self, *args):
        """
        Initializes an MDcontroller instance.

        Parameters
        ----------
        *args : str
            Optional file name to load metadata from.
        """
        if len(args) > 0:
            self.file_name = args[0]
        self.metadata = load_metadata(self.file_name)

    def add_metadata(self, key: str, value: str or dict):
        """
        Adds a key-value pair to the metadata.

        Parameters
        ----------
        key : str
            Key to add.
        value : str or dict
            Value to associate with the key.

        Returns
        -------
        None
        """
        self.metadata[key] = value

    def remove_metadata(self, key: str):
        """
        Removes a key from the metadata.

        Parameters
        ----------
        key : str
            Key to remove.

        Returns
        -------
        None
        """
        self.metadata.pop(key)

    def has_kv(self, key: str, value: str):
        """
        Checks if a key-value pair exists in the metadata.

        Parameters
        ----------
        key : str
            Key to search.
        value : str
            Value to match.

        Returns
        -------
        bool
            True if the key-value pair exists, False otherwise.
        """
        return has_key_value_recur(self.metadata, key, value)

    def get_val(self, key):
        """
        Gets the value of a key in the metadata.

        Parameters
        ----------
        key : str
            Key to search.

        Returns
        -------
        any
            Value associated with the key.
        """
        return get_key_val_recur(key, self.metadata)

    def set_key_val(self, key_name: str, value):
        """
        Sets the value of a key in the metadata.

        Parameters
        ----------
        key_name : str
            Key to set.
        value : any
            Value to associate with the key.

        Returns
        -------
        None
        """
        set_key_val_recur(key_name, value, self.metadata)

    def save_metadata(self, save_path):
        """
        Saves the metadata to a file.

        Parameters
        ----------
        save_path : str
            Path to save the metadata file.

        Returns
        -------
        None
        """
        yaml_file = open(save_path, 'w')
        yaml.safe_dump(self.metadata, yaml_file)


def dict_to_yaml(md_dict: dict, indent=0):
    """
    Converts a dictionary to a YAML-formatted string.

    Parameters
    ----------
    md_dict : dict
        Dictionary to convert.
    indent : int, optional
        Indentation level (default is 0).

    Returns
    -------
    str
        YAML-formatted string.
    """
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
    Recursively checks if a key-value pair exists in a dictionary.

    Parameters
    ----------
    test_dict : dict
        Dictionary to search.
    key : str
        Key to search.
    value : str
        Value to match.

    Returns
    -------
    bool
        True if the key-value pair exists, False otherwise.
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
    Loads metadata from a file.

    Parameters
    ----------
    metadata_file : str
        Path to the metadata file (must be .yaml or .json).

    Returns
    -------
    dict
        Dictionary containing metadata.
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
    Converts a YAML file to a JSON file.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML file.

    Returns
    -------
    str
        Path to the newly created JSON file.
    """
    metadata_dict = load_metadata(yaml_file)
    yaml_pref = yaml_file.split('.')[:-1]
    json_path = yaml_pref + '.json'
    new_json = open(json_path, 'w')
    json.dump(metadata_dict, new_json)
    return json_path


def sniff_metadata(root_dir, target_key, target_value):
    """
    Finds metadata files containing a specific key-value pair.

    Parameters
    ----------
    root_dir : str
        Root directory to search.
    target_key : str
        Key to search for.
    target_value : str
        Value to match.

    Returns
    -------
    list
        List of paths to metadata files containing the key-value pair.
    """
    path_list = []
    for f in os.listdir(root_dir):
        this_path = root_dir + '/' + f
        if os.path.isdir(this_path):
            path_list += sniff_metadata(this_path, target_key, target_value)
        else:
            file_ext = f.split('.')[-1]
            if file_ext == 'yaml' or 'yml':
                mdc = MDcontroller(this_path)
                if mdc.has_kv(target_key, target_value):
                    path_list.append(this_path)
    return path_list