import behavior as tdt


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


class BasicRun:
    """
    A class to represent a basic recording for an experiment. Most applications will extend this class for a specific
    kind of recording (ephys, imaging, behavior, etc.)

    Attributes
    ----------
    data : dict
        data recorded in this run. keys and values are arbitrary and depend on the experiment
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

    def get_key_val(self, key_name):
        return self.info[key_name]

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
        group_out = dict()
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
