import numpy as np
import pandas as pd
import statistics


class Website:
    """Website objects consist of a website number, packets, and various attributes derived from the packets."""

    def __init__(self, website_number: int, packets: list[int], xyz: tuple[int, int, int], sample_size: int):
        """
        The __init__ function is the constructor for a class. It is called when an object of that class
        is instantiated, and it sets up the attributes of that object. In this case, we are setting up
        the attributes for our WebsiteFeatures objects.

        :param self: Refer to the current instance of a class
        :param website_number: int: Identify the website
        :param packets: list[int]: Store the list of packet sizes for a given website
        :return: A dictionary with keys that are strings and values that are tuples of functions
        :doc-author: Trelent
        """
        self.equalized = True
        self.website_number = website_number
        self.packets = packets
        self.percentiles = [10, 25, 50, 75, 90]
        self.max_streaks = (None, None)
        self.sample_size = sample_size
        self.x, self.y, self.z = xyz
        self.features: dict[str:tuple] = {
            "pos_packets": (get_pos_packets, None),
            "neg_packets": (get_neg_packets, None),
            "non_zero_packets": (get_non_zero_packets, None),
            "total_num_pkts_including_zeros": (get_num_pkts_with_zeros, None),
            "total_num_pkts_excluding_zeros": (get_num_pkts_no_zeros, None),
            "total_pkt_size": (get_total_pkt_size, None),
            "total_neg_pkt_size": (get_total_neg_pkt_size, None),
            "total_pos_pkt_size": (get_total_pos_pkt_size, None),
            "avg_pkt_size": (get_avg_pkt_size, None),
            "avg_neg_pkt_size": (get_avg_neg_pkt_size, None),
            "avg_pos_pkt_size": (get_avg_pos_pkt_size, None),
            "count_negative": (get_count_negative, None),
            "count_positive": (get_count_positive, None),
            "count_zeros": (get_count_zeros, None),
            "smallest_pkt_size": (get_smallest_pkt_size, None),
            "largest_pkt_size": (get_largest_pkt_size, None),
            "count_unique_pkt_sizes": (get_count_unique_pkt_sizes, None),
            "standard_dev_total": (get_standard_dev_total, None),
            "standard_dev_neg": (get_standard_dev_neg, None),
            "standard_dev_pos": (get_standard_dev_pos, None),
            "first_x": (get_first_x, None),
            "first_neg_y": (get_first_neg_y, None),
            "first_pos_z": (get_first_pos_z, None),
            "cumsum_all": (get_cumsum_all, None),
            "cumsum_nonzero": (get_cumsum_nonzero, None),
            "cumsum_neg": (get_cumsum_neg, None),
            "cumsum_pos": (get_cumsum_pos, None),
            "highest_neg_streak": (get_highest_neg_streak, None),
            "highest_pos_streak": (get_highest_pos_streak, None),
            "total_pkt_size_percentiles": (get_total_pkt_size_percentiles, None),
            "neg_pkt_size_percentiles": (get_neg_pkt_size_percentiles, None),
            "pos_pkt_size_percentiles": (get_pos_pkt_size_percentiles, None),
            "sample_with_zeros": (get_sample_with_zeros, None),
            "sample_without_zeros": (get_sample_without_zeros, None)
        }

    def get_feature(self, feature: str):
        """
        The get_feature function is a wrapper for the feature's dictionary.
        It takes in a feature name as an argument and returns the value of that feature.
        If it has not been calculated yet, it will call the function associated with that key to calculate it.

        :param self: Represent the instance of the class
        :param feature: str: Specify the feature that is being passed into the function
        :return: The value of the feature
        :doc-author: Trelent
        """

        func, ret = self.features[feature]
        if not ret:
            ret = func(self)
            self.features[feature] = (func, ret)
            return ret
        return ret

    def generate_features(self):
        """
        The generate_features function is a method of the FeatureGenerator class.
        It takes no arguments, and returns nothing. It iterates through all features in the self. Features dictionary,
        and calls get_feature on each one.

        :param self: Represent the instance of the class
        :return: The features of the dataframe
        :doc-author: Trelent
        """

        for feature in self.features.keys():
            self.get_feature(feature)

    def __lt__(self, other):
        # Compare based on the 'number' property
        """
        The __lt__ function is a special function that allows you to compare two objects of the same class.
        In this case, we are comparing two instances of the 'Card' class. The __lt__ function returns True if
        the first object is less than the second object, and False otherwise.

        :param self: Refer to the current instance of the class,
        :param other: Compare the current object to another object
        :return: A boolean value based on the comparison of two objects
        :doc-author: Trelent
        """
        return self.number < other.number

    def __str__(self):
        """
        The __str__ function is called when you print an object. It's a special function that returns a string
        representation of the object. If you don't define it, Python will use its default implementation which is to
        print the type and memory address of the object.

        :param self: Represent the instance of the class
        :return: A string that contains the values of all the attributes
        :doc-author: Trelent
        """
        return f"[{self.website_number}," \
               f"{self.features['avg_pkt_size'][1]}," \
               f"{self.features['avg_neg_pkt_size'][1]}," \
               f"{self.features['avg_pos_pkt_size'][1]}," \
               f"{self.features['total_pkt_size_percentiles'][1][10]}," \
               f"{self.features['total_pkt_size_percentiles'][1][25]}," \
               f"{self.features['total_pkt_size_percentiles'][1][50]}," \
               f"{self.features['total_pkt_size_percentiles'][1][75]}," \
               f"{self.features['total_pkt_size_percentiles'][1][90]}," \
               f"{self.features['neg_pkt_size_percentiles'][1][10]}," \
               f"{self.features['neg_pkt_size_percentiles'][1][25]}," \
               f"{self.features['neg_pkt_size_percentiles'][1][50]}," \
               f"{self.features['neg_pkt_size_percentiles'][1][75]}," \
               f"{self.features['neg_pkt_size_percentiles'][1][90]}," \
               f"{self.features['pos_pkt_size_percentiles'][1][10]}," \
               f"{self.features['pos_pkt_size_percentiles'][1][25]}," \
               f"{self.features['pos_pkt_size_percentiles'][1][50]}," \
               f"{self.features['pos_pkt_size_percentiles'][1][75]}," \
               f"{self.features['pos_pkt_size_percentiles'][1][90]}," \
               f"{self.features['total_pkt_size'][1]}," \
               f"{self.features['total_neg_pkt_size'][1]}," \
               f"{self.features['total_pos_pkt_size'][1]}," \
               f"{self.features['total_num_pkts_including_zeros'][1]}," \
               f"{self.features['count_positive'][1]}," \
               f"{self.features['count_negative'][1]}," \
               f"{self.features['count_zeros'][1]}," \
               f"{self.features['total_num_pkts_excluding_zeros'][1]}," \
               f"{self.features['smallest_pkt_size'][1]}," \
               f"{self.features['largest_pkt_size'][1]}," \
               f"{self.features['count_unique_pkt_sizes'][1]}," \
               f"{self.features['highest_neg_streak'][1]}," \
               f"{self.features['highest_pos_streak'][1]}," \
               f"{self.features['standard_dev_total'][1]}," \
               f"{self.features['standard_dev_neg'][1]}," \
               f"{self.features['standard_dev_pos'][1]}," \
               f"{','.join(map(str, self.features['first_x'][1]))}," \
               f"{','.join(map(str, self.features['first_neg_y'][1]))}," \
               f"{','.join(map(str, self.features['first_pos_z'][1]))}," \
               #f"{','.join(map(str, self.features['sample_with_zeros'][1]))}," \
               #f"{','.join(map(str, self.features['sample_without_zeros'][1]))}]"               
               #f"{','.join(map(str, self.features['cumsum_all'][1]))}," \
               #f"{','.join(map(str, self.features['cumsum_nonzero'][1]))}," \
               #f"{','.join(map(str, self.features['cumsum_neg'][1]))}," \
               #f"{','.join(map(str, self.features['cumsum_pos'][1]))}" \


    @property
    def number(self):
        """
        The number function returns the website number of a given object.

        :param self: Refer to the instance of the class
        :return: The website_number of the object
        :doc-author: Trelent
        """
        return self.website_number

    @property
    def neg_packets(self):
        """
        The neg_packets function returns the number of packets with negative values.


        :param self: Represent the instance of the class
        :return: The number of packets with negative length
        :doc-author: Trelent
        """
        return self.get_feature("neg_packets")

    @property
    def pos_packets(self):
        """
        The pos_packets function returns the number of packets that have been sent by this node.


        :param self: Represent the instance of the class
        :return: The number of packets that are considered positive
        :doc-author: Trelent
        """
        return self.get_feature("pos_packets")

    @property
    def avg_pkt_size(self):
        """
        The avg_pkt_size function returns the average packet size of a flow.

        :param self: Represent the instance of the object that calls this method
        :return: The average packet size
        :doc-author: Trelent
        """
        return self.get_feature("avg_pkt_size")

    @property
    def non_zero_packets(self):
        """
        The non_zero_packets function returns the number of packets that are not zero.

        :param self: Represent the instance of the class
        :return: The number of non-zero packets in the flow
        :doc-author: Trelent
        """
        return self.get_feature("non_zero_packets")

    @property
    def avg_neg_pkt_size(self):
        """
        The avg_neg_pkt_size function returns the average size of negative packets in bytes.


        :param self: Represent the instance of the class
        :return: The average size of packets that are
        :doc-author: Trelent
        """
        return self.get_feature("avg_neg_pkt_size")

    @property
    def avg_pos_pkt_size(self):
        """
        The avg_pos_pkt_size function returns the average size of packets sent by the host.

        :param self: Represent the instance of the class
        :return: The average size of the packets that are positive
        :doc-author: Trelent
        """
        return self.get_feature("avg_pos_pkt_size")

    @property
    def total_pkt_size_percentiles(self):
        """
        The total_pkt_size_percentiles function returns the percentiles of total packet size.


        :param self: Represent the instance of the class
        :return: The percentiles of the total packet size for each flow
        :doc-author: Trelent
        """
        return self.get_feature("total_pkt_size_percentiles")

    @property
    def neg_pkt_size_percentiles(self):
        """
        The neg_pkt_size_percentiles function returns the negative packet size percentiles of a given flow.

        :param self: Represent the instance of the class
        :return: A list of the negative packet size percentiles
        :doc-author: Trelent
        """
        return self.get_feature("neg_pkt_size_percentiles")

    @property
    def pos_pkt_size_percentiles(self):
        """
        The pos_pkt_size_percentiles function returns the percentiles of packet sizes for positive class.
            :return: The percentiles of packet size for positive class.
            :rtype: list

        :param self: Represent instances of a class
        :return: A list of the percentiles for packet size
        :doc-author: Trelent
        """
        return self.get_feature("pos_pkt_size_percentiles")

    @property
    def total_pkt_size(self):
        """
        The total_pkt_size function returns the total size of all packets in bytes.


        :param self: Represent the instance of the class
        :return: The total packet size
        :doc-author: Trelent
        """
        return self.get_feature("total_pkt_size")

    @property
    def total_neg_pkt_size(self):
        """
        The total_neg_pkt_size function returns the total size of all negative packets in bytes.


        :param self: Represent the instance of the class
        :return: The total size of all packets that have a negative packet size
        :doc-author: Trelent
        """
        return self.get_feature("total_neg_pkt_size")

    @property
    def total_pos_pkt_size(self):
        """
        The total_pos_pkt_size function returns the total positive packet size of a given flow.

        :param self: Represent the instance of the class
        :return: The total size of all packets that have a positive packet size
        :doc-author: Trelent
        """
        return self.get_feature("total_pos_pkt_size")

    @property
    def total_num_pkts_including_zeros(self):
        """
        The total_num_pkts_including_zeros function returns the total number of packets including zeros.


        :param self: Represent the instance of the class
        :return: The total number of packets in the flow, including zero-length packets
        :doc-author: Trelent
        """
        return self.get_feature("total_num_pkts_including_zeros")

    @property
    def count_positive(self):
        """
        The count_positive function returns the number of positive words in a text.

        :param self: Refer to the instance of the class
        :return: The number of positive reviews
        :doc-author: Trelent
        """
        return self.get_feature("count_positive")

    @property
    def count_negative(self):
        """
        The count_negative function returns the number of negative words in a tweet.

        :param self: Refer to the object itself
        :return: The number of negative words in the text
        :doc-author: Trelent
        """
        return self.get_feature("count_negative")

    @property
    def count_zeros(self):
        """
        The count_zeros function returns the number of zeros in a given array.

        :param self: Refer to the object itself
        :return: The number of zeros in a given row
        :doc-author: Trelent
        """
        return self.get_feature("count_zeros")

    @property
    def total_num_pkts_excluding_zeros(self):
        """
        The total_num_pkts_excluding_zeros function returns the total number of packets excluding zero-length packets.


        :param self: Represent the instance of the class
        :return: The total number of packets excluding zeros
        :doc-author: Trelent
        """
        return self.get_feature("total_num_pkts_excluding_zeros")

    @property
    def smallest_pkt_size(self):
        """
        The smallest_pkt_size function returns the smallest packet size that can be sent to a device.

        :param self: Represent the instance of the class
        :return: The smallest packet size in bytes
        :doc-author: Trelent
        """
        return self.get_feature("smallest_pkt_size")

    @property
    def largest_pkt_size(self):
        """
        The largest_pkt_size function returns the largest packet size in bytes.


        :param self: Represent the instance of the class
        :return: The largest packet size in the flow
        :doc-author: Trelent
        """
        return self.get_feature("largest_pkt_size")

    @property
    def count_unique_pkt_sizes(self):
        """
        The count_unique_pkt_sizes function returns the number of unique packet sizes in a flow.

        :param self: Represent the instance of the class
        :return: A list of unique packet sizes
        :doc-author: Trelent
        """
        return self.get_feature("count_unique_pkt_sizes")

    @property
    def highest_neg_streak(self):
        """
        The highest_neg_streak function returns the highest negative streak of a given stock.


        :param self: Represent the instance of the class
        :return: The highest negative streak of the player
        :doc-author: Trelent
        """
        return self.get_feature("highest_neg_streak")

    @property
    def highest_pos_streak(self):

        """
        The highest_pos_streak function returns the highest positive streak of a given stock.
            The function takes in a stock object and returns an integer value representing the highest positive streak.

        :param self: Refer to the object itself
        :return: The highest positive streak of the stock
        :doc-author: Trelent
        """
        return self.get_feature("highest_pos_streak")

    @property
    def standard_dev_total(self):
        """
        The standard_dev_total function returns the standard deviation of all values in a given column.

        :param self: Represent the instance of the class
        :return: The standard deviation of the total number of words in each document
        :doc-author: Trelent
        """
        return self.get_feature("standard_dev_total")

    @property
    def standard_dev_neg(self):
        """
        The standard_dev_neg function returns the standard deviation of all negative values in a given column.

        :param self: Allow an object to refer to itself inside a method
        :return: The standard deviation of the negative sentiment scores
        :doc-author: Trelent
        """
        return self.get_feature("standard_dev_neg")

    @property
    def standard_dev_pos(self):
        """
        The standard_dev_pos function returns the standard deviation of the positive values in a given column.

        :param self: Allow an object to refer to itself inside a method
        :return: The standard deviation of the position
        :doc-author: Trelent
        """
        return self.get_feature("standard_dev_pos")

    @property
    def first_x(self):
        """
        The first_x function returns first x number of packet sizes

        :param self: Refer to the instance of the class
        :return: The number of packets
        :doc-author: Trelent
        """
        return self.get_feature("first_x")

    @property
    def first_neg_y(self):
        """
        The first_neg_y function returns the first  y value negative packet sizes

        :param self: Refer to the object itself
        :return: The number of negative packets to return
        :doc-author: Trelent
        """
        return self.get_feature("first_neg_y")

    @property
    def first_pos_z(self):
        """
        The first_pos_z function returns the first z positive packet sizes


        :param self: Allow an object to refer to itself inside a method
        :return: The number of positive packets to return
        :doc-author: Trelent
        """
        return self.get_feature("first_pos_z")

    @property
    def cumsum_all(self):
        """
        The cumsum_all function returns the cumulative sum of all values in a given time series.

        :param self: Refer to the object itself
        :return: The cumulative sum of all the values in the dataframe
        :doc-author: Trelent
        """
        return self.get_feature("cumsum_all")

    @property
    def cumsum_nonzero(self):
        """
        The cumsum_nonzero function returns the cumulative sum of all nonzero values in a given time series.

        :param self: Represent the instance of the class
        :return: The cumulative sum of the nonzero values in the array
        :doc-author: Trelent
        """
        return self.get_feature("cumsum_nonzero")

    @property
    def cumsum_neg(self):
        """
        The cumsum_neg function returns the cumulative sum of negative values in a given time series.

        :param self: Access the features of the dataframe
        :return: The cumulative sum of negative values
        :doc-author: Trelent
        """
        return self.get_feature("cumsum_neg")

    @property
    def cumsum_pos(self):
        """
        The cumsum_pos function returns the cumulative sum of positive values in a time series.

        :param self: Allow an object to refer to itself inside the class
        :return: The cumulative sum of the positive values
        :doc-author: Trelent
        """
        return self.get_feature("cumsum_pos")

    @property
    def sample_with_zeros(self):
        """
        The sample_with_zeros function is used to determine whether the model should sample from a
        zero-inflated distribution. If this function returns True, then the model will sample from a zero-inflated
        distribution. If this function returns False, then the model will not sample from a zero-inflated distribution.

        :param self: Refer to the object itself
        :return: A boolean value
        :doc-author: Trelent
        """
        return self.get_feature("sample_with_zeros")

    @property
    def sample_without_zeros(self):
        """
        The sample_without_zeros function returns a sample of the data without any zeros.
                This is useful for plotting and visualizing the data.

        :param self: Refer to the instance of the class
        :return: A list of all the samples in the dataset that do not contain zeros
        :doc-author: Trelent
        """
        return self.get_feature("sample_without_zeros")


def get_non_zero_packets(website: Website):
    """
    The get_non_zero_packets function takes a Website object as an argument and returns a list of all the non-zero
    packets in that website's packet list.

    :param website: Website: Specify the type of parameter that is expected
    :return: A list of non-zero packets
    :doc-author: Trelent
    """
    return [pkt for pkt in website.packets if pkt != 0]


def get_neg_packets(website: Website):
    """
    The get_neg_packets function takes a Website object as an argument and returns a list of all the negative packets
    in that website's packet_list.


    :param website: Website: Specify the type of object that is passed in
    :return: A list of negative packets
    :doc-author: Trelent
    """
    return [pkt for pkt in website.packets if pkt < 0]


def get_pos_packets(website: Website):
    """
    The get_pos_packets function takes a Website object as an argument and returns a list of positive packets.

    :param website: Website: Specify the type of object that is passed into the function
    :return: A list of positive packets
    :doc-author: Trelent
    """
    return [pkt for pkt in website.packets if pkt > 0]


def get_num_pkts_with_zeros(website: Website):
    """
    The get_num_pkts_with_zeros function takes in a Website object and returns the number of packets that have zero
    bytes. Args: website (Website): A Website object containing information about a website's packets.

    :param website: Website: Specify the type of parameter that is being passed to the function
    :return: The number of packets with zeros in the website
    :doc-author: Trelent
    """
    return len(website.packets)


def get_num_pkts_no_zeros(website: Website):
    """
    The get_num_pkts_no_zeros function takes in a Website object and returns the number of packets that are not zero.
        This is done by using the len() function on the non_zero_packets attribute of a Website object.

    :param website: Website: Specify the website that we want to get the number of packets from
    :return: The number of packets that were not zero
    :doc-author: Trelent
    """
    return len(website.non_zero_packets)


def get_total_pkt_size(website: Website):
    """
    The get_total_pkt_size function takes in a Website object and returns the sum of all packet sizes.
        This function is used to calculate the total size of packets sent by a website.

    :param website: Website: Specify the type of object that is being passed in
    :return: The sum of the absolute values of all the packets in a website
    :doc-author: Trelent
    """
    return sum(abs(num) for num in website.packets)


def get_total_neg_pkt_size(website: Website):
    """
    The get_total_neg_pkt_size function takes in a Website object and returns the sum of all negative packet sizes.
        This function is used to calculate the total number of bytes sent by a website during an attack.

    :param website: Website: Specify the type of object that is being passed into the function
    :return: The sum of all negative packet sizes
    :doc-author: Trelent
    """
    return sum(website.neg_packets)


def get_total_pos_pkt_size(website: Website):
    """
    The get_total_pos_pkt_size function takes in a Website object and returns the sum of all positive packet sizes.
        This function is used to calculate the total number of bytes sent by a website.

    :param website: Website: Specify that the function takes a website object as an argument
    :return: The sum of the positive packets
    :doc-author: Trelent
    """
    return sum(website.pos_packets)


def get_avg_pkt_size(website: Website):
    """
    The get_avg_pkt_size function takes a Website object as an argument and returns the average packet size of that
    website. The function does this by dividing the total number of bytes sent by the number of packets sent.

    :param website: Website: Specify the type of object that is passed into the function
    :return: The average packet size for a website
    :doc-author: Trelent
    """
    return website.total_pkt_size / len(website.packets)


def get_avg_neg_pkt_size(website: Website):
    """
    The get_avg_neg_pkt_size function takes in a Website object and returns the average negative packet size of that
    website. The function first checks to see if there are any negative packets for the given website, and if not,
    it returns 0. If there are negative packets for the given website, then it divides the total_neg_pkt_size by len(
    website.neg_packets) to get an average.

    :param website: Website: Pass in the website object
    :return: The average size of negative packets
    :doc-author: Trelent
    """

    return website.total_neg_pkt_size / website.count_negative


def get_avg_pos_pkt_size(website: Website):
    """
    The get_avg_pos_pkt_size function takes in a Website object and returns the average positive packet size of that
    website. The function does this by dividing the total positive packet size of the website by its number of
    positive packets.

    :param website: Website: Pass the website object to the function
    :return: The average positive packet size of a website
    :doc-author: Trelent
    """
    return website.total_pos_pkt_size / website.count_positive


def get_count_negative(website: Website):
    """
    The get_count_negative function takes in a Website object and returns the number of negative packets that were
    sent to it. Args: website (Website): The website whose negative packet count is being returned.

    :param website: Website: Specify the type of parameter that is being passed into the function
    :return: The number of negative packets in the website
    :doc-author: Trelent
    """
    return len(website.neg_packets)


def get_count_positive(website: Website):
    """
    The get_count_positive function takes in a Website object and returns the number of positive packets
        that were sent to the website.

    :param website: Website: Specify the type of parameter that is expected
    :return: The number of positive packets
    :doc-author: Trelent
    """
    return len(website.pos_packets)


def get_count_zeros(website: Website):
    """
    The get_count_zeros function takes a Website object as an argument and returns the number of zeros in its packets
    list.

    :param website: Website: Pass the website object to the function
    :return: The number of packets with a value of 0
    :doc-author: Trelent
    """
    return website.packets.count(0)


def get_smallest_pkt_size(website: Website):
    """
    The get_smallest_pkt_size function takes in a Website object and returns the smallest packet size of that
    website. The function first finds all non-zero packets, then uses the abs() function to find absolute values of
    each packet, and finally uses min() with key=abs to return the smallest value.

    :param website: Website: Specify the type of object that is being passed to the function
    :return: The smallest packet size in the list of non-zero packets
    :doc-author: Trelent
    """
    return abs(min(website.non_zero_packets, key=abs))


def get_largest_pkt_size(website: Website):
    """
    The get_largest_pkt_size function takes a Website object as an argument and returns the largest packet size
        (in bytes) of all packets sent to or from that website.

    :param website: Website: Specify the type of object that is passed into the function
    :return: The largest packet size of a website
    :doc-author: Trelent
    """
    return abs(max(website.non_zero_packets, key=abs))


def get_count_unique_pkt_sizes(website: Website):
    """
    The get_count_unique_pkt_sizes function takes a Website object as input and returns the number of unique packet
    sizes in that website's packets list.


    :param website: Website: Specify the type of object that is being passed into the function
    :return: The number of unique packet sizes
    :doc-author: Trelent
    """
    return len(set(website.packets))  # includes zeros


def get_standard_dev_total(website: Website):
    """
    The get_standard_dev_total function takes in a Website object and returns the standard deviation of all packets
    in that website. This is done by first converting the list of Packet objects into a pandas DataFrame, then using
    the std() function to calculate the standard deviation.

    :param website: Website: Pass in the website object
    :return: The standard deviation of the total number of packets
    :doc-author: Trelent
    """

    return statistics.stdev(website.packets)


def get_standard_dev_neg(website: Website):
    """
    The get_standard_dev_neg function takes in a Website object and returns the standard deviation of the negative
    packets.


    :param website: Website: Pass the website object to the function
    :return: The standard deviation of the negative packets
    :doc-author: Trelent
    """
    return statistics.stdev(website.neg_packets)


def get_standard_dev_pos(website: Website):
    """
    The get_standard_dev_pos function takes in a Website object and returns the standard deviation of the position of
    all packets sent by that website. This is done by first creating a pandas DataFrame from the pos_packets
    attribute of the given Website object, then using pandas' std() function to calculate its standard deviation.

    :param website: Website: Specify the website we are looking at
    :return: The standard deviation of the number of positive packets for a given website
    :doc-author: Trelent
    """
    return statistics.stdev(website.pos_packets)


def get_first_x(website: Website):
    """
    The get_first_x function takes a Website object and an integer x as input.
    It returns the first x packets of the website.

    :param website: Website: Specify the type of object that is being passed into the function
    :return: The first x packets of the website
    :doc-author: Trelent
    """
    out_list = website.packets[:website.x]
    if len(out_list) < website.x:
        website.equalized = False
    return out_list


def get_first_neg_y(website: Website):
    """
    The get_first_neg_y function takes in a website and an integer y, and returns the first y negative packets of
    that website.


    :param website: Website: Specify that the function is expecting a website object
    :return: The first y negative packets from the website
    :doc-author: Trelent
    """
    out_list = website.neg_packets[:website.y]
    if len(out_list) < website.y:
        website.equalized = False
    return out_list


def get_first_pos_z(website: Website):
    """
    The get_first_pos_z function takes in a website and an integer z, and returns the first z position packets of
    that website.

    :param website: Website: Specify the website object that is being passed into the function
    :return: The first z packets of the website
    :doc-author: Trelent
    """
    out_list =  website.pos_packets[:website.z]
    if len(out_list) < website.z:
        website.equalized = False
    return out_list

def get_cumsum_all(website: Website):
    """
    The get_cumsum_all function takes a website object as an argument and returns a list of the cumulative sum of all
    packets for each time interval. The function uses the get_cumsum function to calculate this value for each time
    interval.

    :param website: Website: Specify the type of parameter that is passed into the function
    :return: The cumulative sum of all the packets
    :doc-author: Trelent
    """
    return [sum(website.packets[:i + 1]) for i in range(len(website.packets))]


def get_cumsum_nonzero(website: Website):
    """
    The get_cumsum_nonzero function takes a Website object as input and returns a list of the cumulative sum of all
    non-zero packets.

    :param website: Website: Specify the website that we want to get the cumsum of nonzero packets for
    :return: A list of cumulative sums
    :doc-author: Trelent
    """
    return [sum(website.non_zero_packets[:i + 1]) for i in range(len(website.non_zero_packets))]


def get_cumsum_neg(website: Website):
    """
    The get_cumsum_neg function takes a website object as an argument and returns a list of the cumulative sum of
    negative packets for that website.

    :param website: Website: Specify the website object that is passed into the function
    :return: A list of the cumulative sum of negative packets
    :doc-author: Trelent
    """
    return [sum(website.neg_packets[:i + 1]) for i in range(len(website.neg_packets))]


def get_cumsum_pos(website: Website):
    """
    The get_cumsum_pos function takes a website object as input and returns a list of the cumulative sum of positive
    packets for that website.

    :param website: Website: Specify the type of object that is being passed to the function
    :return: A list of cumulative sums
    :doc-author: Trelent
    """
    return [sum(website.pos_packets[:i + 1]) for i in range(len(website.pos_packets))]


def get_total_pkt_size_percentiles(website: Website):
    """
    The get_total_pkt_size_percentiles function takes a website object as input and returns the percentiles of the
    total packet size for that website.


    :param website: Website: Specify the website that we want to get the percentiles for
    :return: A list of percentiles
    :doc-author: Trelent
    """
    return get_percentiles(website, website.packets)


def get_neg_pkt_size_percentiles(website: Website):
    """
    The get_neg_pkt_size_percentiles function takes a website object as input and returns the percentiles of negative
    packet sizes. Args: website (Website): A Website object containing information about a specific website.

    :param website: Website: Specify the website that we are looking at
    :return: A list of the percentiles for the negative packet sizes
    :doc-author: Trelent
    """
    return get_percentiles(website, website.neg_packets)


def get_pos_pkt_size_percentiles(website: Website):
    """
    The get_pos_pkt_size_percentiles function takes a website object as input and returns the percentiles of positive
    packet sizes. Args: website (Website): A Website object containing information about a particular website.

    :param website: Website: Specify the website that is being analyzed
    :return: The percentiles of the positive packet sizes
    :doc-author: Trelent
    """
    return get_percentiles(website, website.pos_packets)


def get_highest_neg_streak(website: Website):
    """
    The get_highest_neg_streak function takes in a website object and returns the highest negative streak of that
    website. The function first checks if the value is already stored in the features dictionary, if it is then it
    returns that value. If not, then it calls get_highest_positive_and_negative_streaks to get both values at once
    and stores them into their respective keys before returning the negative streak.

    :param website: Website: Pass in the website object
    :return: The highest negative streak of a website
    :doc-author: Trelent
    """

    func, val = website.features["highest_neg_streak"]
    if not val:
        neg, pos = get_highest_positive_and_negative_streaks(website)
        website.features["highest_neg_streak"] = (get_highest_neg_streak, neg)
        website.features["highest_pos_streak"] = (get_highest_pos_streak, pos)
        return neg
    return val


def get_highest_pos_streak(website: Website):
    """
    The get_highest_pos_streak function returns the highest positive streak of a website.


    :param website: Website: Pass in the website object
    :return: The highest positive streak of a website
    :doc-author: Trelent
    """

    func, val = website.features["highest_pos_streak"]
    if not val:
        neg, pos = get_highest_positive_and_negative_streaks(website)
        website.features["highest_neg_streak"] = (get_highest_neg_streak, neg)
        website.features["highest_pos_streak"] = (get_highest_pos_streak, pos)
        return pos
    return val


def get_highest_positive_and_negative_streaks(website: Website):
    """
    The get_highest_positive_and_negative_streaks function takes in a Website object and returns the highest positive
    and negative streaks of packets. The function iterates through each packet in the website's list of packets,
    keeping track of current positive and negative streaks as well as maximum positive and negative streaks. If a
    packet is less than 0, the current negative streak increases by 1 while the current positive streak resets to 0;
    if it is greater than 0, the opposite occurs. If it is equal to zero, both are reset.

    :param website: Website: Pass in the website object that we want to analyze
    :return: A tuple of the
    :doc-author: Trelent
    """
    max_neg_streak, max_pos_streak = 0, 0
    curr_neg_streak, curr_pos_streak = 0, 0

    for pkt in website.packets:
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


def get_percentiles(website: Website, dataset: list) -> dict:
    """
    The get_percentiles function takes a website object and a dataset as input.
    It returns a dictionary of percentiles, where the keys are the percentiles
    and the values are their corresponding values in the dataset.

    :param website: Website: Access the percentiles attribute of the website class
    :param dataset: list: Pass the dataset to the function
    :return: A dictionary with the percentiles as keys and their values as values
    :doc-author: Trelent
    """
    percentiles_dict = {}
    for percentile in website.percentiles:
        value = np.percentile(dataset, percentile)
        percentiles_dict[percentile] = value
    return percentiles_dict


def get_sample_with_zeros(website: Website):
    """
    The get_sample_with_zeros function takes a Website object as input and returns an array of the same size
    as the website's sample_size attribute. The function accomplishes this by taking every nth packet from the
    website's packets attribute, where n is equal to total_num_pkts / sample_size. This means that if there are
    more zeros than non-zeros in a given website, then more zeros will be included in its sampled data.

    :param website: Website: Pass in the website object
    :return: A sample of the website's packets with zeros added in between them
    :doc-author: Trelent
    """

    step_size = len(website.cumsum_all) / website.sample_size
    steps = np.arange(0, len(website.cumsum_all), step_size)
    out_list = np.interp(steps, np.arange(len(website.cumsum_all)), website.cumsum_all).tolist()
    if len(out_list) < website.sample_size:
        website.equalized = False
    return out_list[:website.sample_size]


def get_sample_without_zeros(website: Website):
    """
    The get_sample_without_zeros function takes a website object as an argument and returns a sample of the non-zero
    packets from that website. The function first calculates the step size, which is equal to the total number of
    packets excluding zeros divided by the sample size. Then it creates an array called steps that contains all
    integers from 1 to the total number of packets excluding zeros with increments equal to step_size. Finally,
    it uses numpy interp function to return a new array containing values interpolated between each pair in steps
    and non_zero_packets.

    :param website: Website: Pass in the website object
    :return: A sample of the non-zero packets
    :doc-author: Trelent
    """

    step_size = len(website.cumsum_nonzero) / website.sample_size
    steps = np.arange(0, len(website.cumsum_nonzero), step_size)
    out_list = np.interp(steps, np.arange(len(website.cumsum_nonzero)), website.cumsum_nonzero).tolist()
    if len(out_list) < website.sample_size:
        website.equalized = False
    return out_list[:website.sample_size]
