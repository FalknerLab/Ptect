import argparse
from gui import PtectApp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TerritoryTools',
                                     description='Data processing library for mouse territorial behaviors',
                                     epilog='See documentation at github.com/Ptect')
    parser.add_argument('datafolder', help='Path to territory run data')
    parser.add_argument('-o', '--outdir', help='Path for saving data')
    parser.add_argument('-gui', action='store_true', help='Pass to enable GUI')
    args = parser.parse_args()
    if args.gui:
        PtectApp(data_folder=args.datafolder)
