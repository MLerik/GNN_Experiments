from pandas.tests.extension.numpy_.test_numpy_nested import np


def graph_plotting_example_1(position: int):
    node = np.zeros(5)
    node[position] = 1
    print('\t\t\t{}'.format(node[4]))
    print('\t\t/')
    print('{}\t{}\t{}\t{}'.format(node[0], node[1], node[2], node[3]))
