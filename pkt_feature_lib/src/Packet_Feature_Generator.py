# !../venv/bin/python
import csv
import json
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from queue import PriorityQueue

import matplotlib.pyplot as plt
from tqdm import tqdm

from Features import *

X, Y, Z = 0, 0, 0


def handle_args():
    """
    The handle_args function is used to parse the command line arguments.
    It takes no parameters and returns an object containing all the command line arguments.
    The function uses argparse, a Python module that makes it easy to write user-friendly
    command-line interfaces.

    :return: A namespace object
    :doc-author: Trelent
    """

    parser = ArgumentParser(description='Turns packet features to CDF graphs')
    parser.add_argument("-i", dest="input_file", required=True, help="(required) input text file", metavar="TXTFILE")
    parser.add_argument('-x', dest="x", default=0, type=int,
                        help="(optional) Add first X number of total packets as features.")
    parser.add_argument('-y', dest="y", default=0, type=int,
                        help="(optional) Add first Y number of negative packets as features.")
    parser.add_argument('-z', dest="z", default=0, type=int,
                        help="(optional) Add first Z number of positive packets as features.")
    parser.add_argument('-csv', dest="csv", action='store_true', help="(optional) Write packet features to a CSV file.")
    parser.add_argument('-cdf', dest="cdfs", action='store_true', help="(optional) Create CDFs from CSV file.")
    parser.add_argument('-ml', dest="ml", action='store_true',
                        help="(optional) Output to text file all websites in the format of websiteNumber1,feature1,"
                             "feature2,...")
    parser.add_argument('-s', dest="s", default=102, type=int, help="(optional) Generate samples using size s.")
    parser.add_argument('--zeros', dest="zeros", choices=["True", "False"],
                        help="(required with -s flag) Specify whether or not to include packets of size zero in the "
                             "sampling.")
    parser.add_argument('-j', dest="num_threads", help="define the number of threads to run with", default=4, type=int)

    args = parser.parse_args()

    return args


def process_file(args, x, y, z) -> PriorityQueue[Website]:
    """
    The process_file function takes in a file path and parses the data into Website objects.
        The function returns a PriorityQueue of Website objects sorted by their priority score.

    :param args: Pass in the command line arguments
    :param x: Value of X
    :param y: Value of Y
    :param z: Value of Z
    :return: A priorityqueue of website objects
    :doc-author: Trelent
    """

    websites = PriorityQueue()
    with open(args.input_file, 'r') as fp:
        for line in tqdm(fp, desc="Parsing Data File Into Website Objects"):
            parsed_line = json.loads(line)
            websites.put(Website(parsed_line[0], parsed_line[1:], (x, y, z), args.s))
    return websites


def work_features(in_list: list[Website]) -> list[Website]:
    """
    The work_features function takes a list of Website objects and generates features for each one.
    It then returns a new list of websites with their features generated.

    :param in_list: list[Website]: Specify the type of the input list
    :return: A new list of websites with their features generated
    :doc-author: Trelent
    """

    result_list = []
    for website in tqdm(in_list, "Generating Features in thread"):
        try:
            website.generate_features()
        except Exception as e:
            print(f"Error generating features for website {website.website_number}: {str(e)}")
        result_list.append(website)

    return result_list


def split(a, n):
    """
    The split function takes a list and splits it into n equal parts.
    If the length of the list is not divisible by n, then the last part will be shorter than all others.

    Parameters:
        a (list): The list to split up.  Must be non-empty!

    :param a: Specify the list to be split
    :param n: Specify the number of sublists to split the list into
    :return: A list of lists
    :doc-author: Trelent
    """
    if n == 0:
        raise ValueError("n cannot be zero")
    if n > len(a):
        raise ValueError("n cannot be greater than the length of a")
    if not a:
        raise ValueError("List 'a' must be non-empty")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def generate_features(websites: list[Website], num_threads: int = 8) -> list[Website]:
    """
    The generate_features function takes in a list of websites and the number of threads to use.
    It then splits the list into num_threads sublists, each containing roughly equal numbers of websites.
    Each thread is given one sublist to work on, and they all run simultaneously using multithreading.
    The results are appended back together into a single list which is returned.

    :param websites: list[Website]: Pass in the websites that need to be processed
    :param num_threads: int: Specify the number of threads to be used in the function
    :return: A list of websites with the features generated
    :doc-author: Trelent
    """
    if not isinstance(websites, list) or not all(isinstance(website, Website) for website in websites):
        raise ValueError("websites must be a list of Website objects")
    if not isinstance(num_threads, int) or num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")

    processed_websites = []

    nested_list = list(split(websites, num_threads))
    [list() for _ in range(num_threads)]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for result in list(executor.map(work_features, nested_list)):
            result_list = list(result)
            processed_websites.extend(result_list)

    return processed_websites


def create_outfile(file_name: str):
    """
    The create_outfile function creates a new empty file.
        If the file already exists, it is overwritten.

        Args:
            file_name (str): The name of the output file to be created.

    :param file_name: str: Specify the name of the file to be created
    :return: A string
    :doc-author: Trelent
    """

    try:
        with open(file_name, 'w+'):
            pass  # This creates an empty file
        print(f"Created a new empty file: {file_name}")
    except FileNotFoundError as e:
        print(f"Error creating a new file: {e}")
        raise
    except PermissionError as e:
        print(f"Error creating a new file: {e}")
        raise


def output_csv(in_list: list[Website]):
    """
    The output_csv function takes in a queue of Website objects and outputs a CSV file

    :param in_list: list[Website]: Specify the type of data that is being passed into the function
    :return: A string that is the file path to the output csv file
    :doc-author: Trelent
    """

    global X, Y, Z
    csv_data = [['Website Number',
                 'Average Packet Size (Abs Value)',
                 'Average Negative Packet Size',
                 'Average Positive Packet Size',
                 'Total Packet Sizes: 10th Percentile',
                 'Total Packet Sizes: 25th Percentile',
                 'Total Packet Sizes: 50th Percentile',
                 'Total Packet Sizes: 75th Percentile',
                 'Total Packet Sizes: 90th Percentile',
                 'Negative Packet Sizes: 10th Percentile',
                 'Negative Packet Sizes: 25th Percentile',
                 'Negative Packet Sizes: 50th Percentile',
                 'Negative Packet Sizes: 75th Percentile',
                 'Negative Packet Sizes: 90th Percentile',
                 'Positive Packet Sizes: 10th Percentile',
                 'Positive Packet Sizes: 25th Percentile',
                 'Positive Packet Sizes: 50th Percentile',
                 'Positive Packet Sizes: 75th Percentile',
                 'Positive Packet Sizes: 90th Percentile',
                 'Total Packet Size (Abs Value)',
                 'Total Negative Packet Size',
                 'Total Positive Packet Size',
                 'Total Number of Packets (including zeros)',
                 'Total Number of Packets (excluding zeros)',
                 'Count of Positive',
                 'Count of Negative',
                 'Count of Zeros',
                 'Smallest Packet Size (Abs Value and excluding zero)',
                 'Largest Packet Size (Abs Value)',
                 'Number of Unique Packets',
                 'Highest Negative Streak',
                 'Highest Positive Streak',
                 'Standard Deviation of Total',
                 'Standard Deviation of Negatives',
                 'Standard Deviation of Positives',
                 f'First {X} Packets',
                 f'First {Y} Negative Packets',
                 f'First {Z} Positive Packets',
                 'Cumulative Sum (including zeros)',
                 'Cumulative Sum (excluding zeros)',
                 'Cumulative Sum Negatives',
                 'Cumulative Sum Positives',
                 'Sample Size',
                 'Sample With Zeros',
                 'Sample Without Zeros']]

    for website in tqdm(in_list, desc="Creating CSV file"):
        csv_data.append([
            website.number, website.avg_pkt_size,
            website.avg_neg_pkt_size,
            website.avg_pos_pkt_size,
            website.total_pkt_size_percentiles[10],
            website.total_pkt_size_percentiles[25],
            website.total_pkt_size_percentiles[50],
            website.total_pkt_size_percentiles[75],
            website.total_pkt_size_percentiles[90],
            website.neg_pkt_size_percentiles[10],
            website.neg_pkt_size_percentiles[25],
            website.neg_pkt_size_percentiles[50],
            website.neg_pkt_size_percentiles[75],
            website.neg_pkt_size_percentiles[90],
            website.pos_pkt_size_percentiles[10],
            website.pos_pkt_size_percentiles[25],
            website.pos_pkt_size_percentiles[50],
            website.pos_pkt_size_percentiles[75],
            website.pos_pkt_size_percentiles[90],
            website.total_pkt_size,
            website.total_neg_pkt_size,
            website.total_pos_pkt_size,
            website.total_num_pkts_including_zeros,
            website.total_num_pkts_excluding_zeros,
            website.count_positive,
            website.count_negative,
            website.count_zeros,
            website.smallest_pkt_size,
            website.largest_pkt_size,
            website.count_unique_pkt_sizes,
            website.features["highest_neg_streak"][1],
            website.features['highest_pos_streak'][1],
            website.standard_dev_total,
            website.standard_dev_neg,
            website.standard_dev_pos,
            website.first_x,
            website.first_neg_y,
            website.first_pos_z,
            website.cumsum_all,
            website.cumsum_nonzero,
            website.cumsum_neg,
            website.cumsum_pos,
            website.sample_size,
            website.sample_with_zeros,
            website.sample_without_zeros])

    file_path = "output.csv"
    with open(file_path, 'w+', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(csv_data)

    return file_path


def create_cdfs(csv_path: str):
    """
    The create_cdfs function generates CDF graphs for each column in the CSV file (except website number).

    :param csv_path: str: Specify the path to the csv file
    :return: Nothing
    :doc-author: Trelent
    """

    df = pd.read_csv(csv_path)

    for column in tqdm(df.columns[1:-3], desc="Generating CDF graphs..."):
        values = df[column].sort_values().reset_index(drop=True)
        cumulative = pd.Series(range(1, len(values) + 1)) / len(values)

        plt.plot(values, cumulative, marker='o')
        plt.xlabel(column)
        plt.ylabel('CDF')
        plt.title(f'CDF for ({column})')
        plt.grid(True)

        save_path = f'../output/cdf_{column}.png'
        plt.savefig(save_path)
        plt.close()


def write_list_to_file(in_list: list[Website]):
    """
    The write_list_to_file function takes a PriorityQueue of Website objects and writes them to an output file. The
    function creates the output file if it does not exist, and overwrites the contents of the file if it does exist.
    The function uses tqdm to display a progress bar while writing each Website object in string form to the output
    file.

    :param in_list: PriorityQueue[Website]: Pass in the priorityqueue of website objects
    :return: Nothing
    :doc-author: Trelent
    """

    file_path = "output/output.ml"

    with open(file_path, 'w+') as fp:
        for website in tqdm(in_list, desc="Writing List to File"):
            fp.write(f"{str(website)}\n")


def equalize_output(websites: list[Website]):
    """
    The equalize_output function takes in a PriorityQueue of Website objects and returns a new PriorityQueue
    of Website objects. The returned queue will contain only those websites that have at least X total packets,
    Y negative packets, and Z positive packets. This is done to ensure that the training data is balanced.

    :param websites: PriorityQueue[Website]: Pass the websites to be equalized into the function
    :return: A queue of websites that have at least x packets, y negative and z positive
    :doc-author: Trelent
    """

    global X, Y, Z
    out_list = []

    for website in tqdm(websites, desc="Equalizing Websites"):
        if website.total_num_pkts_including_zeros >= X and website.count_negative >= Y and website.count_positive >= Z:
            out_list.append(website)

    return out_list


def create_sample_file(websites: list[Website], include_zero: bool = True):
    """
    The create_sample_file function takes in a PriorityQueue of Website objects and an optional boolean value. The
    function then iterates through the queue, creating a list of samples that have more than one sample for each
    website. If the include_zero parameter is set to True, then it will use the total number of packets including
    zeros as its length for sampling purposes. Otherwise, it will use the total number of packets excluding zeros as
    its length for sampling purposes. It then writes these samples to a file

    :param websites: PriorityQueue[Website]: Pass in a queue of websites
    :param include_zero: bool: Determine whether to include zero packets in the sample
    :return: A list of samples
    :doc-author: Trelent
    """

    samples = []

    for website in tqdm(websites, desc="Collecting Websites to Sample"):

        if include_zero:
            length = website.total_num_pkts_including_zeros

        else:
            length = website.total_num_pkts_excluding_zeros

        if length < website.sample_size:
            continue

        samples.append(website)

    counts = defaultdict(int)
    for sample in tqdm(samples, desc="Sampling Websites"):
        counts[str(sample.website_number)] += 1

    samples = [sample for sample in tqdm(samples, desc="Filtering Samples") if counts[str(sample.website_number)] > 1]

    path = "samples.txt"
    create_outfile(path)
    with open(path, "w+") as fp:
        for sample in tqdm(samples, desc="Writing Samples to file"):
            if include_zero:
                samples = sample.sample_with_zeros
            else:
                samples = sample.sample_without_zeros

            fp.write(f"{sample.website_number},{','.join(map(str, (round(value) for value in samples)))}\n")


def main():
    """
    The main function of the program.

    :doc-author: Trelent
    """

    args = handle_args()

    websites = process_file(args, args.x, args.y, args.z)
    web_list = []

    while websites.qsize() != 0:
        web_list.append(websites.get())
        websites.task_done()

    web_features = generate_features(web_list, args.num_threads)

    if args.csv:
        output_csv(web_features)

    if args.cdfs:
        csv_path = output_csv(web_features)
        create_cdfs(csv_path)

    if args.ml:
        web_equalized = equalize_output(web_features)
        write_list_to_file(web_equalized)

    if args.s:
        create_sample_file(web_features)


if __name__ == "__main__":
    main()
