import ast
import numpy as np
from typing import List, Tuple
from collections import defaultdict

#--------------- DIFFERENT FROM NATHANS ---------------#
#set this to False if you don't want to include the 0's found in the arrays in size_collected.txt
include_zeros = False
#sample data pulled from size_collected.txt
if include_zeros:
    cumsum_outfile = "/Users/enodynowski/fingerprinting/cumulative_sum.txt"
    sampling_outfile = "/Users/enodynowski/fingerprinting/sampled.txt"
else:
    cumsum_outfile = "/Users/enodynowski/fingerprinting/cumulative_sum_no_zeros.txt"
    sampling_outfile = "/Users/enodynowski/fingerprinting/sampled_no_zeros.txt"


#remove an item from the list
def remove_items(test_list, item):
    res = [i for i in test_list if i != item]
    return res


#my code to make a cumulative sum
def Cumulative(lists):
    cu_list = []
    key = lists[0]
    if include_zeros:
        length = len(lists)
        cu_list = [sum(lists[1:x:1]) for x in range(0, length+1)]
    else:
        list_no_zeros = remove_items(lists, 0)
        length = len(list_no_zeros)
        cu_list = [sum(list_no_zeros[1:x:1]) for x in range(0, length+1)]
    cu_list[1] = key
    return cu_list[1:]



#iterate through the lines in the size_collected file and calculate cumulative sum for each.
def calcCumSum():
    with open("/Users/enodynowski/fingerprinting/size_collected.txt", "r") as in_file:
        with open(cumsum_outfile, "w") as cumsum_outfile_write:
            for line in in_file:
                #name the file appropriately for if 0's are being included
                #read the line into an array
                int_list = ast.literal_eval(line)
                #write the cumulative sum to the outfile
                cumsum_outfile_write.write(str(Cumulative(int_list)) + '\n')
        cumsum_outfile_write.close()
    in_file.close()
calcCumSum()
#--------------- NATHAN'S SAMPLING CODE ---------------#


def format_output(results: List[Tuple[float, List[float], List[float]]]) -> List[str]:
    return ["[" + str(key) + "," + ','.join(map(str, (round(value) for value in sample))) + "]" for key, steps, sample in results]

def evenly_distributed_sample_from_file(input_file_path, sample_size: int) -> List[Tuple[float, List[float], List[float]]]:
    results = []  
    with open(input_file_path, "r") as input_file:
        input_file.seek(0)
        for line in input_file:
            # Convert line into list
            values = ast.literal_eval(line.strip())
            # The first value is the key
            key = values[0]

            # The remaining values form the array
            line_no_key = values[1:]
            # check array lengths vs sample size
            if len(line_no_key) < sample_size:
                continue
            
            # Compute the length of the array
            length = len(line_no_key)
            
            # Compute the step size
            step_size = length / sample_size

            #print(step_size)
            # Compute the steps
            steps = np.arange(1, length, step_size).tolist()
            #print(steps)
            # Sample the array using linear interpolation
            # x = steps
            # xp = np.arrange(length)
            # fp = array
            sample = np.interp(steps, np.arange(length), line_no_key).tolist()
            # Remove first non-key value
            del sample[0]
            # Append the key, steps and the sample to the results
            results.append((key, steps, sample))
    input_file.close()
    return results

def write_samples_to_file(samples, output_file_path):
    with open(output_file_path, "w") as sampling_outfile_write:
        for sample in samples:
            sampling_outfile_write.write(sample + "\n")
    sampling_outfile_write.close


def group_and_filter_by_key(samples: List[Tuple[float, List[float], List[float]]]) -> List[Tuple[float, List[float], List[float]]]:
# Create a dictionary where the keys are the sample keys and the values are the counts of samples with that key
    counts = defaultdict(int)
    for sample in samples:
        counts[sample[0]] += 1

# Filter out samples whose key is associated with a count of less than 2
    filtered_samples = [sample for sample in samples if counts[sample[0]] >= 1]
    return filtered_samples


results = evenly_distributed_sample_from_file(cumsum_outfile, 102)
filtered_results = group_and_filter_by_key(results)
formatted_samples = format_output(filtered_results)
write_samples_to_file(formatted_samples, sampling_outfile)
