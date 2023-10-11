#!../venv/bin/python

import sys
from argparse import ArgumentParser
from Features import Website
from ast import literal_eval as line_to_array
from tqdm import tqdm
from threading import Thread, Lock
from queue import PriorityQueue
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

X, Y, Z = 0, 0, 0


def handle_args():
    """
    The handle_args function is used to parse the command line arguments.
    It takes no parameters and returns an object containing all of the command line arguments.
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

    global X, Y, Z
    X, Y, Z = int(args.x), int(args.y), int(args.z)

    # if args.s > 0 and not args.zeros:
    #     parser.error("-s flag requires the use of --zeros flag")
    #     sys.exit(-1)
    # if args.zeros and not args.s:
    #     parser.error("--zeros flag requires the use of -s flag")
    #     sys.exit(-1)

    return args


def process_file(args) -> PriorityQueue[Website]:
    global X, Y, Z
    websites = PriorityQueue()
    with open(args.input_file, 'r') as fp:
        for line in tqdm(fp.readlines(), desc="Parsing Data File Into Website Objects"):
            parsed_line = line_to_array(line)
            websites.put(Website(parsed_line[0], parsed_line[1:], (X, Y, Z), args.s))
    return websites


def work_features(in_list: list[Website], out_list: list[Website]):
    for website in tqdm(in_list, "Generating Features in thread"):
        website.generate_features()
        out_list.append(website)


def work_samples(in_queue: PriorityQueue[Website], out_queue: PriorityQueue[Website], lock: Lock, pbar):
    while not in_queue.empty():
        website = in_queue.get()
        website.generate_samples()

        if in_queue.qsize() % 1000 == 0:
            with lock:
                pbar.update(1)

        out_queue.put(website)
        in_queue.task_done()


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def generate_features(websites: PriorityQueue[Website], num_threads: int = 8) -> PriorityQueue[Website]:
    processed_websites = PriorityQueue()

    website_list = []
    for website in range(websites.qsize()):
        website_list.append(websites.get())

    nested_list = list(split(website_list, num_threads))
    result_lists = [[] for _ in range(num_threads)]
    threads = []

    for i in range(num_threads):
        thread = Thread(target=work_features, args=(nested_list[i], result_lists[i]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    result_list = [item for sublist in result_lists for item in sublist]
    for website in result_list:
        processed_websites.put(website)

    return processed_websites


def create_outfile(file_name: str):
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Deleted the existing file: {file_name}")

    try:
        with open(file_name, 'w') as new_file:
            pass  # This creates an empty file
        print(f"Created a new empty file: {file_name}")
    except Exception as e:
        print(f"Error creating a new file: {e}")


def output_csv(in_queue: PriorityQueue[Website]):
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
    pbar = tqdm(total=in_queue.qsize(), desc="Creating CSV file")

    for _ in range(in_queue.qsize()):
        website = in_queue.get()
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
        in_queue.task_done()
        pbar.update(1)
    pbar.close()

    file_path = "../output/output.csv"
    create_outfile(file_path)
    with open(file_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(csv_data)

    return file_path


def create_cdfs(csv_path: str):
    """Generates CDF graphs for each column in the CSV file (except website number)"""
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


def write_list_to_file(in_queue: PriorityQueue[Website]):
    file_path = "../output/output.ml"
    create_outfile(file_path)
    pbar = tqdm(total=in_queue.qsize(), desc="Writing List to File")

    with open(file_path, 'w') as fp:
        for _ in range(in_queue.qsize()):
            website = in_queue.get()
            fp.write(f"{str(website)}\n")
            in_queue.task_done()
            pbar.update(1)


def equalize_output(websites: PriorityQueue[Website]):
    global X, Y, Z
    out_queue = PriorityQueue()
    pbar = tqdm(total=websites.qsize(), desc="Equalizing Websites")

    for _ in range(websites.qsize()):
        website = websites.get()
        if website.total_num_pkts_including_zeros >= X and website.count_negative >= Y and website.count_positive >= Z:
            out_queue.put(website)
        pbar.update(1)
        websites.task_done()
    return out_queue


def create_sample_file(websites: PriorityQueue[Website]):
    pass


def main():
    args = handle_args()
    websites = process_file(args)
    web_features = generate_features(websites, args.num_threads)

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
