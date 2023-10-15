import pytest
from random import randrange

from ..src.Packet_Feature_Generator import *


@pytest.fixture
def website():
    with open("../data/small.txt") as fp:
        lines = fp.readlines()
        parsed_line = json.loads(lines[randrange(len(lines))])
        return Website(parsed_line[0], parsed_line[1:], (25, 25, 25), 102)


def test_get_non_zero_packets(website):
    webs: Website = website
    packets = webs.packets
    scrubbed = list(filter(lambda x: x != 0, packets))
    assert scrubbed == get_non_zero_packets(website)


def test_get_neg_packets(website):
    packets = website.packets
    scrubbed = list(filter(lambda x: x < 0, packets))
    assert scrubbed == get_neg_packets(website)


def test_get_pos_packets(website):
    packets = website.packets
    scrubbed = list(filter(lambda x: x > 0, packets))
    assert scrubbed == get_pos_packets(website)


def test_get_num_pkts_with_zeros(website):
    assert len(website.packets) == get_num_pkts_with_zeros(website)


def test_get_num_pkts_no_zeros(website):
    assert len(list(filter(lambda x: x != 0, website.packets))) == get_num_pkts_no_zeros(website)


def test_get_total_pkt_size(website):
    total = 0
    for i in website.packets:
        if i < 0:
            i *= -1
        total += i

    assert total == get_total_pkt_size(website)


def test_get_total_neg_pkt_size(website):
    total = 0
    for i in website.packets:
        if i < 0:
            total += i

    assert total == get_total_neg_pkt_size(website)


def test_get_total_pos_pkt_size(website):
    total = 0
    for i in website.packets:
        if i > 0:
            total += i

    assert total == get_total_pos_pkt_size(website)


def test_get_avg_pkt_size(website):
    avg = get_total_pkt_size(website) / len(website.packets)
    assert avg == get_avg_pkt_size(website)


def test_get_avg_neg_pkt_size(website):
    avg = get_total_neg_pkt_size(website) / len(get_neg_packets(website))
    assert avg == get_avg_neg_pkt_size(website)


def test_get_avg_pos_pkt_size(website):
    avg = get_total_pos_pkt_size(website) / len(get_pos_packets(website))
    assert avg == get_avg_pos_pkt_size(website)


def test_get_count_negative(website):
    assert len(get_neg_packets(website)) == get_count_negative(website)


def test_get_count_positive(website):
    assert len(get_pos_packets(website)) == get_count_positive(website)


def test_get_count_zeros(website):
    count = 0
    for i in website.packets:
        if i == 0:
            count += 1

    assert count == get_count_zeros(website)


def test_get_smallest_pkt_size(website):
    smallest = 9999999999999
    for i in website.non_zero_packets:
        i = abs(i)
        if i < smallest:
            smallest = i

    assert smallest == get_smallest_pkt_size(website)


def test_get_largest_pkt_size(website):
    largest = 0

    for i in website.non_zero_packets:
        i = abs(i)
        if i > largest:
            largest = i

    assert largest == get_largest_pkt_size(website)


def test_get_count_unique_pkt_sizes(website):
    uniques = []
    for i in website.packets:
        if i not in uniques:
            uniques.append(i)

    assert len(uniques) == get_count_unique_pkt_sizes(website)


def test_get_standard_dev_total(website):
    assert round(pd.DataFrame(website.packets).std().values[0], 2) == round(get_standard_dev_total(website), 2)


def test_get_standard_dev_neg(website):
    assert round(pd.DataFrame(website.neg_packets).std().values[0], 2) == round(get_standard_dev_neg(website), 2)


def test_get_standard_dev_pos(website):
    assert round(pd.DataFrame(website.pos_packets).std().values[0], 2) == round(get_standard_dev_pos(website), 2)


def test_get_first_x(website):
    items = []

    for i in range(website.x):
        if i < len(website.packets):
            items.append(website.packets[i])
        else:
            break

    assert items == get_first_x(website)


def test_get_first_neg_y(website):
    items = []
    for i in range(website.x):
        if i < len(website.neg_packets):
            items.append(website.neg_packets[i])
        else:
            break

    assert items == get_first_neg_y(website)


def test_get_first_pos_z(website):
    items = []
    for i in range(website.x):
        if i < len(website.neg_packets):
            items.append(website.pos_packets[i])
        else:
            break

    assert items == get_first_pos_z(website)


def test_get_cumsum_all(website):
    cum_sum = 0
    cum_sum_list = []

    for num in website.packets:
        cum_sum += num
        cum_sum_list.append(cum_sum)

    assert cum_sum_list == get_cumsum_all(website)


def test_get_cumsum_nonzero(website):
    cum_sum = 0
    cum_sum_list = []

    for num in website.non_zero_packets:
        cum_sum += num
        cum_sum_list.append(cum_sum)

    assert cum_sum_list == get_cumsum_nonzero(website)


def test_get_cumsum_neg(website):
    cum_sum = 0
    cum_sum_list = []

    for num in website.neg_packets:
        cum_sum += num
        cum_sum_list.append(cum_sum)

    assert cum_sum_list == get_cumsum_neg(website)


def test_get_cumsum_pos(website):
    cum_sum = 0
    cum_sum_list = []

    for num in website.pos_packets:
        cum_sum += num
        cum_sum_list.append(cum_sum)

    assert cum_sum_list == get_cumsum_pos(website)


def calculate_percentile(data, percentile):
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    sorted_data = sorted(data)
    n = len(sorted_data)
    rank = (percentile / 100) * (n - 1)

    if rank.is_integer():
        return sorted_data[int(rank)]

    k = int(rank)
    d = rank - k
    return sorted_data[k] + d * (sorted_data[k + 1] - sorted_data[k])


def test_get_percentiles(website):
    perc_dict = {}
    for percentile in website.percentiles:
        perc_dict[percentile] = calculate_percentile(website.packets, percentile)

    assert perc_dict == get_percentiles(website, website.packets)


def test_get_total_pkt_size_percentiles(website):
    perc_dict = {}
    for percentile in website.percentiles:
        perc_dict[percentile] = calculate_percentile(website.packets, percentile)

    assert perc_dict == get_total_pkt_size_percentiles(website)


def test_get_neg_pkt_size_percentiles(website):
    perc_dict = {}
    for percentile in website.percentiles:
        perc_dict[percentile] = calculate_percentile(website.neg_packets, percentile)

    assert perc_dict == get_neg_pkt_size_percentiles(website)


def test_get_pos_pkt_size_percentiles(website):
    perc_dict = {}
    for percentile in website.percentiles:
        perc_dict[percentile] = calculate_percentile(website.pos_packets, percentile)

    assert perc_dict == get_pos_pkt_size_percentiles(website)


def test_get_highest_positive_and_negative_streaks(website):
    current_pos_streak = 0
    max_pos_streak = 0
    current_neg_streak = 0
    max_neg_streak = 0

    for num in website.packets:
        if num > 0:
            current_pos_streak += 1
            if current_pos_streak > max_pos_streak:
                max_pos_streak = current_pos_streak
        else:
            current_pos_streak = 0

    for num in website.packets:
        if num < 0:
            current_neg_streak += 1
            if current_neg_streak > max_neg_streak:
                max_neg_streak = current_neg_streak
        else:
            current_neg_streak = 0

    neg, pos = get_highest_positive_and_negative_streaks(website)

    assert max_pos_streak == pos
    assert max_neg_streak == neg


def test_website_properties(website):
    assert website.number == website.website_number
    assert website.neg_packets == get_neg_packets(website)
    assert website.pos_packets == get_pos_packets(website)
    assert website.avg_pkt_size == get_avg_pkt_size(website)
    assert website.non_zero_packets == get_non_zero_packets(website)
    assert website.avg_neg_pkt_size == get_avg_neg_pkt_size(website)
    assert website.avg_pos_pkt_size == get_avg_pos_pkt_size(website)
    assert website.total_pkt_size_percentiles == get_total_pkt_size_percentiles(website)
    assert website.neg_pkt_size_percentiles == get_neg_pkt_size_percentiles(website)
    assert website.pos_pkt_size_percentiles == get_pos_pkt_size_percentiles(website)
    assert website.total_pkt_size == get_total_pkt_size(website)
    assert website.total_pos_pkt_size == get_total_pos_pkt_size(website)
    assert website.total_neg_pkt_size == get_total_neg_pkt_size(website)
    assert website.total_num_pkts_including_zeros == get_num_pkts_with_zeros(website)
    assert website.count_positive == get_count_positive(website)
    assert website.count_negative == get_count_negative(website)
    assert website.count_zeros == get_count_zeros(website)
    assert website.total_num_pkts_excluding_zeros == get_num_pkts_no_zeros(website)
    assert website.smallest_pkt_size == get_smallest_pkt_size(website)
    assert website.largest_pkt_size == get_largest_pkt_size(website)
    assert website.count_unique_pkt_sizes == get_count_unique_pkt_sizes(website)
    assert website.highest_neg_streak == get_highest_neg_streak(website)
    assert website.highest_pos_streak == get_highest_pos_streak(website)
    assert website.standard_dev_total == get_standard_dev_total(website)
    assert website.standard_dev_neg == get_standard_dev_neg(website)
    assert website.standard_dev_pos == get_standard_dev_pos(website)
    assert website.first_x == get_first_x(website)
    assert website.first_neg_y == get_first_neg_y(website)
    assert website.first_pos_z == get_first_pos_z(website)
    assert website.cumsum_all == get_cumsum_all(website)
    assert website.cumsum_nonzero == get_cumsum_nonzero(website)
    assert website.cumsum_neg == get_cumsum_neg(website)
    assert website.cumsum_pos == get_cumsum_pos(website)
    assert website.sample_with_zeros == get_sample_with_zeros(website)
    assert website.sample_without_zeros == get_sample_without_zeros(website)


def test_validate_features(website):
    for key, tup in website.features.items():
        assert True == isinstance(tup, tuple)
        assert len(tup) == 2
        assert tup[0] is not None


def test_validate_generate_features(website):
    website.generate_features()
    for key, val in website.features.items():
        assert True == isinstance(val, tuple)
        assert val[0] is not None
        assert val[1] is not None


# I highly suggest that you implement some tests here that check that the correct string is getting generated
# As of now we know that most of your functions are working correclty however you are not sure they are coming out
# as expected to be written into a file.

class TestGetHighestPositiveAndNegativeStreaks:

    #  Test with a website object that has only positive packets
    def test_positive_packets(self):
        website = Website(1, [1, 2, 3, 4, 5], (1, 2, 3), 5)
        result = get_highest_positive_and_negative_streaks(website)
        assert result == (0, 5)

    #  Test with a website object that has only negative packets
    def test_negative_packets(self):
        website = Website(1, [-1, -2, -3, -4, -5], (1, 2, 3), 5)
        result = get_highest_positive_and_negative_streaks(website)
        assert result == (5, 0)

    #  Test with a website object that has both positive and negative packets
    def test_mixed_packets(self):
        website = Website(1, [1, -2, 3, -4, 5], (1, 2, 3), 5)
        result = get_highest_positive_and_negative_streaks(website)
        assert result == (1, 1)

    #  Test with a website object that has one packet
    def test_single_packet(self):
        website = Website(1, [1], (1, 2, 3), 1)
        result = get_highest_positive_and_negative_streaks(website)
        assert result == (0, 1)


