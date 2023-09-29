from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

class Website():
    """Website objects consist of a website number, packets, and various attributes derived from the packets."""

    def __init__(self, number, packets):
        self._number = number
        self._packets = packets
        self._percentiles = [10, 25, 50, 75, 90]
        self._non_zero_packets = [pkt for pkt in packets if pkt != 0]
        self._neg_packets = [pkt for pkt in packets if pkt < 0]
        self._pos_packets = [pkt for pkt in packets if pkt > 0]

    def get_features(self, x:int, y:int, z:int):
        # static features
        self._total_num_pkts_including_zeros = len(self._packets)
        self._total_num_pkts_excluding_zeros = len(self._non_zero_packets)

        self._total_pkt_size = sum(abs(num) for num in self._packets)
        self._total_neg_pkt_size = sum(self._neg_packets)
        self._total_pos_pkt_size = sum(self._pos_packets)

        self._avg_pkt_size = self._total_pkt_size / len(self._packets)
        self._avg_neg_pkt_size = self._total_neg_pkt_size / len(self._neg_packets)
        self._avg_pos_pkt_size = self._total_pos_pkt_size / len(self._pos_packets)

        self._count_negative = len(self._neg_packets)
        self._count_positive = len(self._pos_packets)
        self._count_zeros = self._packets.count(0)

        self._smallest_pkt_size = abs(min(self._non_zero_packets, key=abs))
        self._largest_pkt_size = abs(max(self._non_zero_packets, key=abs))
        self._count_unique_pkt_sizes = len(set(self._packets)) # includes zeros

        self._standard_dev_total = (pd.DataFrame(self._packets)).std().values[0]
        self._standard_dev_neg = (pd.DataFrame(self._neg_packets)).std().values[0]
        self._standard_dev_pos = (pd.DataFrame(self._pos_packets)).std().values[0]

        self._first_x = self._packets[:x]
        self._first_neg_y = self._neg_packets[:y]
        self._first_pos_z = self._pos_packets[:z]
        

        # dynamic features
        self._highest_neg_streak, self._highest_pos_streak = self.get_highest_positive_and_negative_streaks()
        self._total_pkt_size_percentiles = self.get_percentiles(self._packets) # doesn't do absolute value
        self._neg_pkt_size_percentiles = self.get_percentiles(self._neg_packets)
        self._pos_pkt_size_percentiles = self.get_percentiles(self._pos_packets)


    def get_highest_positive_and_negative_streaks(self):
        max_neg_streak, max_pos_streak = 0, 0
        curr_neg_streak, curr_pos_streak = 0, 0

        for pkt in self._packets:
            if pkt < 0:
                curr_neg_streak += 1
                curr_pos_streak = 0
                max_neg_streak = max(max_neg_streak, curr_neg_streak)
            elif pkt > 0:
                curr_pos_streak += 1
                curr_neg_streak = 0
                max_pos_streak = max(max_pos_streak, curr_pos_streak)
            else:
                curr_neg_streak = 0
                curr_pos_streak = 0

        return max_neg_streak, max_pos_streak

    def get_percentiles(self, dataset:list) -> dict:
            percentiles_dict = {}
            for percentile in self._percentiles:
                value = np.percentile(dataset, percentile)
                percentiles_dict[percentile] = value

            return percentiles_dict


    def __str__(self):
        return f"{self._number},{self._avg_pkt_size},{self._avg_neg_pkt_size},{self._avg_pos_pkt_size}," + \
            f"{self._total_pkt_size_percentiles[10]},{self._total_pkt_size_percentiles[25]},{self._total_pkt_size_percentiles[50]}," + \
            f"{self._total_pkt_size_percentiles[75]},{self._total_pkt_size_percentiles[90]}," + \
            f"{self._neg_pkt_size_percentiles[10]},{self._neg_pkt_size_percentiles[25]},{self._neg_pkt_size_percentiles[50]}," + \
            f"{self._neg_pkt_size_percentiles[75]},{self._neg_pkt_size_percentiles[90]}," + \
            f"{self._pos_pkt_size_percentiles[10]},{self._pos_pkt_size_percentiles[25]},{self._pos_pkt_size_percentiles[50]}," + \
            f"{self._pos_pkt_size_percentiles[75]},{self._pos_pkt_size_percentiles[90]}," + \
            f"{self._total_pkt_size},{self._total_neg_pkt_size},{self._total_pos_pkt_size}," + \
            f"{self.total_num_pkts_including_zeros},{self._count_positive},{self._count_negative},{self._count_zeros},{self._total_num_pkts_excluding_zeros}," + \
            f"{self._smallest_pkt_size},{self._largest_pkt_size},{self._count_unique_pkt_sizes}," + \
            f"{self._highest_neg_streak},{self._highest_pos_streak}," + \
            f"{self._standard_dev_total},{self._standard_dev_neg},{self._standard_dev_pos}," + \
            f"{','.join(map(str, self._first_x))},{','.join(map(str, self._first_neg_y))},{','.join(map(str, self.first_pos_z))}"


    @property
    def number(self):
        return self._number
    @property
    def avg_pkt_size(self):
        return self._avg_pkt_size
    @property
    def avg_neg_pkt_size(self):
        return self._avg_neg_pkt_size
    @property
    def avg_pos_pkt_size(self):
        return self._avg_pos_pkt_size
    @property
    def total_pkt_size_percentiles(self):
        return self._total_pkt_size_percentiles
    @property
    def neg_pkt_size_percentiles(self):
        return self._neg_pkt_size_percentiles
    @property
    def pos_pkt_size_percentiles(self):
        return self._pos_pkt_size_percentiles
    @property
    def total_pkt_size(self):
        return self._total_pkt_size
    @property
    def total_neg_pkt_size(self):
        return self._total_neg_pkt_size
    @property
    def total_pos_pkt_size(self):
        return self._total_pos_pkt_size
    @property
    def total_num_pkts_including_zeros(self):
        return self._total_num_pkts_including_zeros
    @property
    def count_positive(self):
        return self._count_positive
    @property
    def count_negative(self):
        return self._count_negative
    @property
    def count_zeros(self):
        return self._count_zeros
    @property
    def total_num_pkts_excluding_zeros(self):
        return self._total_num_pkts_excluding_zeros
    @property
    def smallest_pkt_size(self):
        return self._smallest_pkt_size
    @property
    def largest_pkt_size(self):
        return self._largest_pkt_size
    @property
    def count_unique_pkt_sizes(self):
        return self._count_unique_pkt_sizes
    @property
    def highest_neg_streak(self):
        return self._highest_neg_streak
    @property
    def highest_pos_streak(self):
        return self._highest_pos_streak
    @property
    def standard_dev_total(self):
        return self._standard_dev_total
    @property
    def standard_dev_neg(self):
        return self._standard_dev_neg
    @property
    def standard_dev_pos(self):
        return self._standard_dev_pos    
    @property
    def first_x(self):
        return self._first_x
    @property
    def first_neg_y(self):
        return self._first_neg_y
    @property
    def first_pos_z(self):
        return self._first_pos_z


def equalize_output(websites:list, x:int, y:int, z:int):
    """Filters out any websites where first_x, first_neg_y, or first_pos_z are less than x,y,z respectively to ensure each website has equal number of features."""
    filtered_websites = [website for website in websites if 
                         website.total_num_pkts_including_zeros >= x and
                         website.count_negative >= y and
                         website.count_positive >= z]
    return filtered_websites


def create_ml_file(websites:list):
    with open('ml.txt', 'w') as file:
        for website in tqdm(websites, desc="Creating ml.txt file..."):
            print(website, file=file)


def create_cdfs(csv:str):
    """Generates CDF graphs for each column in the CSV file (except website number)"""
    df = pd.read_csv(csv)

    for column in tqdm(df.columns[1:-3], desc="Generating CDF graphs..."):
        values = df[column].sort_values().reset_index(drop=True)
        cumulative = pd.Series(range(1, len(values) + 1)) / len(values)
        
        plt.plot(values, cumulative, marker='o')
        plt.xlabel(column)
        plt.ylabel('CDF')
        plt.title(f'CDF for ({column})')
        plt.grid(True)

        # show the graphs
        #plt.show()

        # save the graphs
        save_path = f'cdf_{column}.png'
        plt.savefig(save_path)
        plt.close()

def output_csv(outfile:str, websites:list, x:int, y:int, z:int):
    """Writes packet features to a CSV."""

    csv_data = [['Website Number', 'Average Packet Size (Abs Value)', 'Average Negative Packet Size', 'Average Positive Packet Size', \
                'Total Packet Sizes: 10th Percentile', 'Total Packet Sizes: 25th Percentile', 'Total Packet Sizes: 50th Percentile', \
                'Total Packet Sizes: 75th Percentile', 'Total Packet Sizes: 90th Percentile', \
                'Negative Packet Sizes: 10th Percentile', 'Negative Packet Sizes: 25th Percentile', 'Negative Packet Sizes: 50th Percentile', \
                'Negative Packet Sizes: 75th Percentile', 'Negative Packet Sizes: 90th Percentile', \
                'Positive Packet Sizes: 10th Percentile', 'Positive Packet Sizes: 25th Percentile', 'Positive Packet Sizes: 50th Percentile', \
                'Positive Packet Sizes: 75th Percentile', 'Positive Packet Sizes: 90th Percentile', \
                'Total Packet Size (Abs Value)', 'Total Negative Packet Size', 'Total Positive Packet Size', \
                'Total Number of Packets (including zeros)', 'Total Number of Packets (excluding zeros)', \
                'Count of Positive', 'Count of Negative', 'Count of Zeros', \
                'Smallest Packet Size (Abs Value and excluding zero)', 'Largest Packet Size (Abs Value)', \
                'Number of Unique Packets', 'Highest Negative Streak', 'Highest Positive Streak', \
                'Standard Deviation of Total', 'Standard Deviation of Negatives', 'Standard Deviation of Positives', \
                f'First {x} Packets', f'First {y} Negative Packets', f'First {z} Positive Packets']]
    
    for website in tqdm(websites, desc="Creating CSV file..."):
        csv_data.append([website.number, website.avg_pkt_size, website.avg_neg_pkt_size, website.avg_pos_pkt_size, \
                        website.total_pkt_size_percentiles[10], website.total_pkt_size_percentiles[25], website.total_pkt_size_percentiles[50], \
                        website.total_pkt_size_percentiles[75], website.total_pkt_size_percentiles[90], \
                        website.neg_pkt_size_percentiles[10], website.neg_pkt_size_percentiles[25], website.neg_pkt_size_percentiles[50], \
                        website.neg_pkt_size_percentiles[75], website.neg_pkt_size_percentiles[90], \
                        website.pos_pkt_size_percentiles[10], website.pos_pkt_size_percentiles[25], website.pos_pkt_size_percentiles[50], \
                        website.pos_pkt_size_percentiles[75], website.pos_pkt_size_percentiles[90], \
                        website.total_pkt_size, website.total_neg_pkt_size, website.total_pos_pkt_size, \
                        website.total_num_pkts_including_zeros, website.total_num_pkts_excluding_zeros, \
                        website.count_positive, website.count_negative, website.count_zeros, website.smallest_pkt_size, website.largest_pkt_size, \
                        website.count_unique_pkt_sizes, website.highest_neg_streak, website.highest_pos_streak, \
                        website.standard_dev_total, website.standard_dev_neg, website.standard_dev_pos, \
                        website.first_x, website.first_neg_y, website.first_pos_z])

    with open(outfile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
    
    return outfile

def process_file(infile:str):
    """Returns a list of comma-separated numbers for each website line."""
    with open(infile, 'r') as file:
        lines = file.readlines()

    websites = []
    for line in lines:
        line = line.strip()[1:-1]  # Remove the square brackets and leading/trailing spaces
        websites.append(list(map(int, line.split(','))))  # Convert comma-separated numbers into a list
    
    return [Website(website[0], website[1:]) for website in websites] # Initialize list of Website objects

def handle_args():
    """Ensures input and output files are passed in as argument."""

    parser = ArgumentParser(description='Turns packet features to CDF graphs')
    parser.add_argument("-i", dest="input_file", required=True, help="(required) input text file", metavar="TXTFILE")
    parser.add_argument("-o", dest="output_file", required=True, help="(required) output CSV file", metavar="CSVFILE")
    parser.add_argument('-x', dest="x", default=0, type=int, help="(optional) Add first X number of total packets as features.")
    parser.add_argument('-y', dest="y", default=0, type=int, help="(optional) Add first Y number of negative packets as features.")
    parser.add_argument('-z', dest="z", default=0, type=int, help="(optional) Add first Z number of positive packets as features.")
    parser.add_argument('-cdf', dest="cdfs", action='store_true', help="(optional) Create CDFs from CSV file.")
    parser.add_argument('-ml', dest="ml", action='store_true', help="(optional) Output to text file all websites in the format of websiteNumber1,feature1,feature2,...")


    return parser.parse_args()


def main() -> int:
    """Parses input textfile, derives website packet features, outputs to CSV, and optionally autogenerates CDFs."""
    args = handle_args()
    websites = process_file(args.input_file)

    for website in tqdm(websites, desc="Gathering packet attributes for each website..."):
        website.get_features(args.x, args.y, args.z)

    csv = output_csv(args.output_file, websites, args.x, args.y, args.z)

    if args.cdfs:
        create_cdfs(csv)

    if args.ml:
        filtered_websites = equalize_output(websites, args.x, args.y, args.z)
        create_ml_file(filtered_websites)

    return 0

if __name__ == "__main__":
    main()
