import os
import numpy as np
import pickle

# Parameters for Weibull distribution
shape_param = 3
scale_param = 45
max_item_size = 100
bin_capacity = 100

np.random.seed(1234)

def generate_weibull_instances(num_instances, num_items, shape, scale, max_size):
    instances = []
    for _ in range(num_instances):
        # Sampling from Weibull distribution
        samples = np.random.weibull(shape, num_items) * scale

        # Clipping and rounding
        items = np.clip(samples, None, max_size)
        items = np.round(items).astype(int)

        instances.append(items)
    return instances


def l1_bound(items: tuple[int, ...], capacity: int) -> float:
  """Computes L1 lower bound on OPT for bin packing.

  Args:
    items: Tuple of items to pack into bins.
    capacity: Capacity of bins.

  Returns:
    Lower bound on number of bins required to pack items.
  """
  return np.ceil(np.sum(items) / capacity)


def l1_bound_dataset(instances: dict) -> float:
  """Computes the mean L1 lower bound across a dataset of bin packing instances.

  Args:
    instances: Dictionary containing a set of bin packing instances.

  Returns:
    Average L1 lower bound on number of bins required to pack items.
  """
  l1_bounds = []
  for name in instances:
    instance = instances[name]
    l1_bounds.append(l1_bound(instance['items'], instance['capacity']))
  return np.mean(l1_bounds)

def generate_datasets():
    basepath = os.path.dirname(__file__)
    os.makedirs(os.path.join(basepath, "dataset"), exist_ok=True)

    # Generating datasets
    training_data = generate_weibull_instances(1, 5000, shape_param, scale_param, max_item_size)
    training_data_1k = generate_weibull_instances(1, 1000, shape_param, scale_param, max_item_size)
    validation_data = generate_weibull_instances(5, 5000, shape_param, scale_param, max_item_size)
    validation_data_1k = generate_weibull_instances(5, 1000, shape_param, scale_param, max_item_size)
    test_data_1k100 = generate_weibull_instances(5, 1000, shape_param, scale_param, max_item_size)
    test_data_5k100 = generate_weibull_instances(5, 5000, shape_param, scale_param, max_item_size)
    test_data_10k100 = generate_weibull_instances(5, 10000, shape_param, scale_param, max_item_size)
    test_data_1k500 = generate_weibull_instances(5, 1000, shape_param, scale_param, max_item_size)
    test_data_5k500 = generate_weibull_instances(5, 5000, shape_param, scale_param, max_item_size)
    test_data_10k500 = generate_weibull_instances(5, 10000, shape_param, scale_param, max_item_size)

    # Saving datasets as pickle files, e.g {train_i: {capacity: 100, num_items: 5000, items: [1, 2, 3, ...]},...}
    weibull_5k_train1 = {'train_' + str(i): {'capacity': 100, 'num_items': len(training_data[i]), 'items': training_data[i]} for i in range(len(training_data))}
    weibull_5k_train2 = {'train_' + str(i): {'capacity': 500, 'num_items': len(training_data[i]), 'items': training_data[i]} for i in range(len(training_data))}
    weibull_1k_train1 = {'train_' + str(i): {'capacity': 100, 'num_items': len(training_data_1k[i]), 'items': training_data_1k[i]} for i in range(len(training_data_1k))}
    weibull_1k_train2 = {'train_' + str(i): {'capacity': 500, 'num_items': len(training_data_1k[i]), 'items': training_data_1k[i]} for i in range(len(training_data_1k))}
    weibull_5k_val1 = {'val_' + str(i): {'capacity': 100, 'num_items': len(validation_data[i]), 'items': validation_data[i]} for i in range(len(validation_data))}
    weibull_5k_val2 = {'val_' + str(i): {'capacity': 500, 'num_items': len(validation_data[i]), 'items': validation_data[i]} for i in range(len(validation_data))}
    weibull_1k_val1 = {'val_' + str(i): {'capacity': 100, 'num_items': len(validation_data_1k[i]), 'items': validation_data_1k[i]} for i in range(len(validation_data_1k))}
    weibull_1k_val2 = {'val_' + str(i): {'capacity': 500, 'num_items': len(validation_data_1k[i]), 'items': validation_data_1k[i]} for i in range(len(validation_data_1k))}
    weibull_1k_test100 = {'test_' + str(i): {'capacity': 100, 'num_items': len(test_data_1k100[i]), 'items': test_data_1k100[i]} for i in range(len(test_data_1k100))}
    weibull_5k_test100 = {'test_' + str(i): {'capacity': 100, 'num_items': len(test_data_5k100[i]), 'items': test_data_5k100[i]} for i in range(len(test_data_5k100))}
    weibull_10k_test100 = {'test_' + str(i): {'capacity': 100, 'num_items': len(test_data_10k100[i]), 'items': test_data_10k100[i]} for i in range(len(test_data_10k100))}
    weibull_1k_test500 = {'test_' + str(i): {'capacity': 500, 'num_items': len(test_data_1k500[i]), 'items': test_data_1k500[i]} for i in range(len(test_data_1k500))}
    weibull_5k_test500 = {'test_' + str(i): {'capacity': 500, 'num_items': len(test_data_5k500[i]), 'items': test_data_5k500[i]} for i in range(len(test_data_5k500))}
    weibull_10k_test500 = {'test_' + str(i): {'capacity': 500, 'num_items': len(test_data_10k500[i]), 'items': test_data_10k500[i]} for i in range(len(test_data_10k500))}

    # Note that weibull_5k_test is provided by Romera-Paredes et al. (https://github.com/google-deepmind/funsearch/blob/main/bin_packing/bin_packing.ipynb).

    # Add l1_bound to each dataset
    weibull_5k_train1['l1_bound'] = l1_bound_dataset(weibull_5k_train1)
    weibull_5k_train2['l1_bound'] = l1_bound_dataset(weibull_5k_train2)
    weibull_1k_train1['l1_bound'] = l1_bound_dataset(weibull_1k_train1)
    weibull_1k_train2['l1_bound'] = l1_bound_dataset(weibull_1k_train2)
    weibull_5k_val1['l1_bound'] = l1_bound_dataset(weibull_5k_val1)
    weibull_5k_val2['l1_bound'] = l1_bound_dataset(weibull_5k_val2)
    weibull_1k_val1['l1_bound'] = l1_bound_dataset(weibull_1k_val1)
    weibull_1k_val2['l1_bound'] = l1_bound_dataset(weibull_1k_val2)
    weibull_1k_test100['l1_bound'] = l1_bound_dataset(weibull_1k_test100)
    weibull_5k_test100['l1_bound'] = l1_bound_dataset(weibull_5k_test100)
    weibull_10k_test100['l1_bound'] = l1_bound_dataset(weibull_10k_test100)
    weibull_1k_test500['l1_bound'] = l1_bound_dataset(weibull_1k_test500)
    weibull_5k_test500['l1_bound'] = l1_bound_dataset(weibull_5k_test500)
    weibull_10k_test500['l1_bound'] = l1_bound_dataset(weibull_10k_test500)

    print(weibull_5k_train1['l1_bound'], weibull_5k_train2['l1_bound'], weibull_1k_train1['l1_bound'], weibull_1k_train2['l1_bound'])
    print(weibull_5k_val1['l1_bound'], weibull_5k_val2['l1_bound'], weibull_1k_val1['l1_bound'], weibull_1k_val2['l1_bound'])
    print(weibull_1k_test100['l1_bound'])
    print(weibull_5k_test100['l1_bound'])
    print(weibull_10k_test100['l1_bound'])
    print(weibull_1k_test500['l1_bound'])
    print(weibull_5k_test500['l1_bound'])
    print(weibull_10k_test500['l1_bound'])


    # Saving datasets as pickle files
    pickle.dump(weibull_5k_train1, open(os.path.join(basepath, 'dataset/weibull_5k_train1.pickle'), 'wb'))
    pickle.dump(weibull_5k_train2, open(os.path.join(basepath, 'dataset/weibull_5k_train2.pickle'), 'wb'))
    pickle.dump(weibull_1k_train1, open(os.path.join(basepath, 'dataset/weibull_1k_train1.pickle'), 'wb'))
    pickle.dump(weibull_1k_train2, open(os.path.join(basepath, 'dataset/weibull_1k_train2.pickle'), 'wb'))
    pickle.dump(weibull_5k_val1, open(os.path.join(basepath,'dataset/weibull_5k_val1.pickle'), 'wb'))
    pickle.dump(weibull_5k_val2, open(os.path.join(basepath,'dataset/weibull_5k_val2.pickle'), 'wb'))
    pickle.dump(weibull_1k_val1, open(os.path.join(basepath,'dataset/weibull_1k_val1.pickle'), 'wb'))
    pickle.dump(weibull_1k_val2, open(os.path.join(basepath,'dataset/weibull_1k_val2.pickle'), 'wb'))
    pickle.dump(weibull_1k_test100, open(os.path.join(basepath,'dataset/weibull_1k_test_100.pickle'), 'wb'))
    pickle.dump(weibull_5k_test100, open(os.path.join(basepath,'dataset/weibull_5k_test_100.pickle'), 'wb'))
    pickle.dump(weibull_10k_test100, open(os.path.join(basepath,'dataset/weibull_10k_test_100.pickle'), 'wb'))
    pickle.dump(weibull_1k_test500, open(os.path.join(basepath,'dataset/weibull_1k_test_500.pickle'), 'wb'))
    pickle.dump(weibull_5k_test500, open(os.path.join(basepath,'dataset/weibull_5k_test_500.pickle'), 'wb'))
    pickle.dump(weibull_10k_test500, open(os.path.join(basepath,'dataset/weibull_10k_test_500.pickle'), 'wb'))

if __name__ == "__main__":
    generate_datasets()