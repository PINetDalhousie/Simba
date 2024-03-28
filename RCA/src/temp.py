# Test hypothesis
import numpy as np

NUM_BS = 5
LENGTH = 1000
FAILURE_TYPES = 3
FAILURE_RATE = 0.02
TEST_SIZE = 0.2
NUM_RUNS = 1000



def method1(class_counts, failure_rate):
    """
    10% of the time (length 100) fails and then distribute randomly
    """
    # Choose failure rate percentage of random numbers between 0 and 99
    failure_indices = np.random.choice(LENGTH, int(failure_rate*LENGTH), replace=False)

    # Assign each failure index an integer between 1 and 3 inclusive
    failure_types = np.random.randint(1, FAILURE_TYPES+1, size=len(failure_indices))

    # Make tuples of failure_indices and failure_types
    failure_tuples = list(zip(failure_indices, failure_types))

    # Set the corresponding indices in class_counts to the corresponding failure type
    # Choose a random index between 0 and NUM_BS-1
    for tuple in failure_tuples:
        class_counts[np.random.randint(0, NUM_BS), tuple[0]] = tuple[1]

    return class_counts


def method2(class_counts, failure_rate):
    """
    We first pick a BS randomly. Then for that BS, failure_rate*LENGTH of the time, fails.
    """
    # Iterate over the BS
    for bs in range(NUM_BS):
        # Choose failure rate percentage of random numbers between 0 and 99
        failure_indices = np.random.choice(LENGTH, int(failure_rate*LENGTH), replace=True)

        # Assign each failure index an integer between 1 and 3 inclusive
        failure_types = np.random.randint(1, FAILURE_TYPES+1, size=len(failure_indices))

        # Make tuples of failure_indices and failure_types
        failure_tuples = list(zip(failure_indices, failure_types))

        # Set the corresponding indices in class_counts to the corresponding failure type
        for tuple in failure_tuples:
            class_counts[bs, tuple[0]] = tuple[1]

    return class_counts
    


def calc_class_counts_stats(class_counts, time_range:float):
    """
    Calculate the number of occurances of each failure type in the last time_range
    percentage of time.
    """
    # Calculate the number of samples in the last time_range
    last_time_range = class_counts[:,-int(time_range*LENGTH):]
    
    # Calculate how many times 1,2,3 occur in last_time_range
    failure_rate_dict = {i: (np.count_nonzero(last_time_range == i),np.count_nonzero(last_time_range == i)/last_time_range.size)  for i in range(1, FAILURE_TYPES+1)}
    
    return failure_rate_dict
    

# Create NUM_BS lists of zeros, each of length LENGTH
class_counts = np.zeros((NUM_BS, LENGTH))

average_of_runs_dict = {i:(0,0) for i in range(1, FAILURE_TYPES+1)}
for i in range(NUM_RUNS):
    class_counts = method1(class_counts, failure_rate=FAILURE_RATE*NUM_BS)
    failure_rate_dict = calc_class_counts_stats(class_counts, time_range=TEST_SIZE)
    for key in failure_rate_dict:
        average_of_runs_dict[key] = (average_of_runs_dict[key][0]+failure_rate_dict[key][0], average_of_runs_dict[key][1]+failure_rate_dict[key][1])

for key in average_of_runs_dict:
    average_of_runs_dict[key] = (average_of_runs_dict[key][0]/NUM_RUNS, average_of_runs_dict[key][1]/NUM_RUNS)

print(average_of_runs_dict)




# Create NUM_BS lists of zeros, each of length LENGTH
class_counts = np.zeros((NUM_BS, LENGTH))

average_of_runs_dict = {i:(0,0) for i in range(1, FAILURE_TYPES+1)}
for i in range(NUM_RUNS):
    class_counts = method2(class_counts, failure_rate=FAILURE_RATE)
    failure_rate_dict = calc_class_counts_stats(class_counts, time_range=TEST_SIZE)
    for key in failure_rate_dict:
        average_of_runs_dict[key] = (average_of_runs_dict[key][0]+failure_rate_dict[key][0], average_of_runs_dict[key][1]+failure_rate_dict[key][1])

for key in average_of_runs_dict:
    average_of_runs_dict[key] = (average_of_runs_dict[key][0]/NUM_RUNS, average_of_runs_dict[key][1]/NUM_RUNS)

print(average_of_runs_dict)
