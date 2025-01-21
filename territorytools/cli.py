import argparse
from territorytools.gui import PtectApp
from territorytools.demo import run_demo



def main():
    parser = argparse.ArgumentParser(prog='TerritoryTools',
                                     description='Data processing library for mouse territorial behaviors',
                                     epilog='See documentation at github.com/Ptect')
    parser.add_argument('-folder', help='Path to territory run data')
    parser.add_argument('-o', '--outdir', help='Path for saving data')
    parser.add_argument('-gui', action='store_true', help='Pass to enable GUI')
    parser.add_argument('-demo', action='store_true', help='Pass to run demo')
    args = parser.parse_args()
    print(parser)
    # if args.demo:
    #     run_demo()
    # else:
    #     if args.gui:
    #         app = PtectApp(data_folder=args.datafolder)



if __name__ == '__main__':
    main()
