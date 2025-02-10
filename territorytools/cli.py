import os
import argparse
from territorytools.gui import PtectApp
from territorytools.demo import run_demo
from territorytools.process import process_all_data



def main():
    """
    Main function to parse arguments and execute the appropriate action.

    Returns
    -------
    None
    """
    flags = ['-folder', '-o', '-gui', '-demo']
    defaults = [None, None, False, False]
    parser = argparse.ArgumentParser(prog='TerritoryTools',
                                     description='Data processing library for mouse territorial behaviors',
                                     epilog='See documentation at github.com/Ptect')
    parser.add_argument(flags[0], help='Path to territory run data')
    parser.add_argument(flags[1], '--outdir', help='Path for saving data')
    parser.add_argument(flags[2], action='store_true', help='Pass to enable GUI')
    parser.add_argument(flags[3], action='store_true', help='Pass to run demo')
    args = vars(parser.parse_args())
    num_args = 0
    for k, d in zip(args.keys(), defaults):
        if args[k] != d:
            num_args += 1

    if num_args == 0:
        print_info()

    if args['demo']:
        run_demo()
    elif args['gui']:
        data_fold=None
        if args['folder'] is not None:
            data_fold = args['folder']
        app = PtectApp(data_folder=data_fold)
    elif args['folder'] is not None:
        out_path = None
        if args['outdir'] is not None:
            out_path = args['outdir']
        print(f'Processing {args['folder']}...')
        process_all_data(args['folder'], out_path=out_path, skip_ptect=False)
    else:
        print('No data folder path provided')


def print_info():
    """
    Prints the version information from the version file.

    Returns
    -------
    None
    """
    v_file = open(os.path.abspath('resources/version.txt'), 'r')
    print(v_file.read())


if __name__ == '__main__':
    main()
