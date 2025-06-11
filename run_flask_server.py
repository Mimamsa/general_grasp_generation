"""Initial entire service
Auther: Jason
Date: 2019/02/12
"""

import os, sys, argparse, traceback
from flask_utils.log import Log
from flask_utils.server import Server


def main(args):

    Log.set(level=args.level, dashboard=args.dashboard)

    # Flask API version
    try:
        Server.run()
    except (EOFError, KeyboardInterrupt):
        Log.warning("Terminated")
    except:
        Log.warning('Unknown exit')
        traceback.print_exc()
    finally:
        Log.warning("End all thread - Exit program")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-l','--level', type=str, choices=['DEBUG','INFO','WARNING','ERROR'],
    help='Choose the logging level like DEBUG or WARING', default='DEBUG')
    parser.add_argument('-d','--dashboard', type=bool, choices=[True,False],
    help='Enable the dashboard mode', default=False)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
