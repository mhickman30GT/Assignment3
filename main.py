import argparse
import datetime
import json
import os
import shutil
import multiprocessing

import problem as pro

# GLOBAL VARIABLES
TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PATH = os.path.dirname(os.path.realpath(__file__))
CNFG = os.path.join(os.path.join(PATH, "data"), "config.json")


def get_args():
    """ Process args from command line """
    parser = argparse.ArgumentParser()

    # Set data to run
    parser.add_argument(
        "-d", "--data",
    )

    # Set part to run
    parser.add_argument(
        "-p", "--part",
    )

    # Set type of run
    parser.add_argument(
        "-t", "--type",
    )

    # Set optimizer to tune
    parser.add_argument(
        "-c", "--cluster",
    )

    # Set hyper param to tune
    parser.add_argument(
        "-r", "--reducer",
    )

    # Set name of the experiment
    parser.add_argument(
        "-n", "--name", default=f'RUN_DATA_{TIME}',
    )
    return parser.parse_args()


def main():
    """ Main """
    # Process command line
    args = get_args()

    # Process directories and config
    dir_name = os.path.join(os.path.join(PATH, "out"), args.name)
    os.makedirs(dir_name)
    shutil.copy(CNFG, dir_name)

    # Open config
    with open(CNFG, "r") as open_file:
        jsonconfig = json.load(open_file)
    config = jsonconfig[args.data]

    # Create output directory
    out_dir = os.path.join(dir_name, args.name)
    os.makedirs(out_dir)

    # Initialize data class
    dataset = pro.DataSet(args.data, config["label"],
                          os.path.join(os.path.join(PATH, "data"), config["file"]))
    dataset.process()

    # Parse the requested problems
    if args.cluster:
        clust = args.cluster
    else:
        clust = None

    if args.reducer:
        reducer = args.reducer
    else:
        reducer = None

    # Core count for problem tuning
    core_count = round(multiprocessing.cpu_count() * .75)

    # Initialize the experiment
    exp = pro.ExperimentClass(args.name, dataset, args.part, args.type,
                              clust, reducer, config, out_dir)

    # Run the experiment
    exp.run()


if __name__ == "__main__":
    main()
