# import pytest
import pytest

from ..src.Packet_Feature_Generator import *
from random import randrange
import pytest, os

@pytest.fixture
def website():
    with open("../data/small.txt") as fp:
        lines = fp.readlines()
        parsed_line = json.loads(lines[randrange(len(lines))])
        return Website(parsed_line[0], parsed_line[1:], (25, 25, 25), 102)

class TestProcessFile:

    #  process a file with valid input and return a list of Website objects
    def test_valid_input_return_list_of_website_objects(self):
        args = Namespace(input_file="valid_data.txt", num_threads=4, x=1, y=2, z=3, s=102)
        websites = process_file(args)
        assert isinstance(websites, list)
        assert all(isinstance(website, Website) for website in websites)

    #  process a file with no data and return an empty list
    def test_no_data_return_empty_list(self):
        args = Namespace(input_file="empty_data.txt", num_threads=4, x=1, y=2, z=3, s=102)
        websites = process_file(args)
        assert isinstance(websites, list)
        assert len(websites) == 0

    #  process a file with one line of data and return a list with one Website object
    def test_one_line_of_data_return_list_with_one_website_object(self):
        args = Namespace(input_file="one_line_data.txt", num_threads=4, x=1, y=2, z=3, s=102)
        websites = process_file(args)
        assert isinstance(websites, list)
        assert len(websites) == 1
        assert isinstance(websites[0], Website)

    #  process a file with invalid input and raise an exception
    def test_invalid_input_raise_exception(self):
        args = Namespace(input_file="invalid_data.txt", num_threads=4, x=1, y=2, z=3)
        with pytest.raises(Exception):
            process_file(args)

    #  process a file with missing input file path and raise an exception
    def test_missing_input_file_path_raise_exception(self):
        args = Namespace(num_threads=4, x=1, y=2, z=3)
        with pytest.raises(Exception):
            process_file(args)

    #  process a file with missing xyz coordinates and raise an exception
    def test_missing_xyz_coordinates_raise_exception(self):
        args = Namespace(input_file="valid_data.txt", num_threads=4)
        with pytest.raises(Exception):
            process_file(args)


class TestWorkFeatures:

    #  Generates features for a website object
    def test_generate_features(self, website):
        result = work_features(website)
        assert result.features != {}

    #  Returns the website object
    def test_return_website_object(self, website):
        result = work_features(website)
        assert isinstance(result, Website)

    #  Handles normal input data
    def test_handles_normal_input_data(self, website):
        result = work_features(website)
        assert result.features['avg_pkt_size'][1]

    #  Handles website object with no packets
    def test_handles_no_packets(self, website):
        result = work_features(website)
        assert result.features['avg_pkt_size'][1]



# Generated by CodiumAI

import pytest

class TestGenerateFeatures:

    #  Generates features for a list of Website objects
    def test_generate_features_for_list_of_websites(self, website):
        websites = [
            website,
            website,
            website,
            website,
            website
        ]
        num_threads = 4

        processed_websites = generate_features(websites, num_threads)

        assert len(processed_websites) == len(websites)
        for website in processed_websites:
            assert isinstance(website, Website)
            assert website.features != {}
            assert website.features['avg_pkt_size'][1]
            assert website.features['avg_neg_pkt_size'][1]
            assert website.features['avg_pos_pkt_size'][1]
            assert website.features['total_pkt_size'][1]
            assert website.features['total_neg_pkt_size'][1]
            assert website.features['total_pos_pkt_size'][1]
            assert website.features['total_num_pkts_including_zeros'][1]
            assert website.features['count_positive'][1]
            assert website.features['count_negative'][1]
            assert website.features['count_zeros'][1]
            assert website.features['total_num_pkts_excluding_zeros'][1]
            assert website.features['smallest_pkt_size'][1]
            assert website.features['largest_pkt_size'][1]
            assert website.features['count_unique_pkt_sizes'][1]
            assert website.features['highest_neg_streak'][1]
            assert website.features['highest_pos_streak'][1]
            assert website.features['standard_dev_total'][1]
            assert website.features['standard_dev_neg'][1]
            assert website.features['standard_dev_pos'][1]
            assert website.features['first_x'][1]
            assert website.features['first_neg_y'][1]
            assert website.features['first_pos_z'][1]
            assert website.features['cumsum_all'][1]
            assert website.features['cumsum_nonzero'][1]
            assert website.features['cumsum_neg'][1]
            assert website.features['cumsum_pos'][1]
            ...

    #  Uses multithreading to speed up the process
    def test_generate_features_with_multithreading(self, website):
        websites = [
            website,
            website,
            website,
            website,
            website
        ]
        num_threads = 4

        processed_websites = generate_features(websites, num_threads)

        assert len(processed_websites) == len(websites)
        for website in processed_websites:
            assert isinstance(website, Website)
            assert website.features != {}
            ...

class TestCreateOutfile:

    #  Creates a new empty file with the specified name
    def test_create_new_empty_file(self):
        file_name = "test_file.txt"
        create_outfile(file_name)
        assert os.path.exists(file_name)
        assert os.stat(file_name).st_size == 0
        os.remove(file_name)

    #  Overwrites an existing file with the specified name
    def test_overwrite_existing_file(self):
        file_name = "test_file.txt"
        with open(file_name, "w") as fp:
            fp.write("This is an existing file")
        create_outfile(file_name)
        assert os.path.exists(file_name)
        assert os.stat(file_name).st_size == 0
        os.remove(file_name)

    #  Raises FileNotFoundError if the specified directory does not exist
    def test_directory_not_exist(self):
        file_name = "nonexistent_directory/test_file.txt"
        with pytest.raises(FileNotFoundError):
            create_outfile(file_name)

    #  Raises PermissionError if the user does not have permission to create a file in the specified directory
    def test_permission_error(self):
        file_name = "/root/test_file.txt"
        with pytest.raises(PermissionError):
            create_outfile(file_name)

    #  The function should return a string indicating the success of the file creation
    def test_return_string(self):
        file_name = "test_file.txt"
        assert create_outfile(file_name) == file_name
        os.remove(file_name)

    #  The function should handle file names with special characters
    def test_special_characters(self):
        file_name = "test_file!@#$%^&*().txt"
        create_outfile(file_name)
        assert os.path.exists(file_name)
        assert os.stat(file_name).st_size == 0
        os.remove(file_name)

