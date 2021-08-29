import json

import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


def non_iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle, shuffle_digits=False):
    assert (nb_nodes > 0 and nb_nodes <= 10)

    digits = torch.arange(10) if shuffle_digits == False else torch.randperm(10,
                                                                             generator=torch.Generator().manual_seed(0))

    # split the digits in a fair way
    digits_split = list()
    i = 0
    for n in range(nb_nodes, 0, -1):
        inc = int((10 - i) / n)
        digits_split.append(digits[i:i + inc])
        i += inc

    # load and shuffle nb_nodes*n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=nb_nodes * n_samples_per_node,
                                         shuffle=shuffle)
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = dataiter.next()

    data_splitted = list()
    for i in range(nb_nodes):
        idx = torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(
            0).bool()  # get indices for the digits
        data_splitted.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images_train_mnist[idx], labels_train_mnist[idx]), batch_size=batch_size,
            shuffle=shuffle))

    return data_splitted


def iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle):
    # load and shuffle n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=n_samples_per_node,
                                         shuffle=shuffle)
    dataiter = iter(loader)

    data_splitted = list()
    for _ in range(nb_nodes):
        data_splitted.append(
            torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(dataiter.next())), batch_size=batch_size,
                                        shuffle=shuffle))

    return data_splitted


def get_MNIST(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, n_test=1, batch_size=25, shuffle=True):
    dataset_loaded_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    dataset_loaded_test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    if type == "iid":
        train = iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
        test = iid_split(dataset_loaded_test, n_test, n_samples_test, batch_size, shuffle)
    elif type == "non_iid":
        train = non_iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
        test = non_iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
    else:
        train = []
        test = []

    return train, test


def plot_samples(data, channel: int, title=None, plot_name="", n_examples=20):
    n_rows = int(n_examples / 5)
    plt.figure(figsize=(1 * n_rows, 1 * n_rows))
    if title: plt.suptitle(title)
    X, y = data
    for idx in range(n_examples):
        ax = plt.subplot(n_rows, 5, idx + 1)

        image = 255 - X[idx, channel].view((28, 28))
        ax.imshow(image, cmap='gist_gray')
        ax.axis("off")

    if plot_name != "": plt.savefig(f"plots/" + plot_name + ".png")
    plt.tight_layout()


def convert(arr):
    l = []
    for a in arr:
        for b in a:
            for c in b:
                l.append(c)
    return l


if __name__ == "__main__":
    mnist_iid_train_dls, mnist_iid_test_dls = get_MNIST("iid",
                                                        n_samples_train=60,
                                                        n_samples_test=6,
                                                        n_clients=1000,
                                                        n_test=1000,
                                                        batch_size=60,
                                                        shuffle=True)

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    i = 0
    train_path = 'data/MNIST/iid/train/all_data_0_niid_0_keep_10_train_9.json'
    for train_i in mnist_iid_train_dls:
        uname = 'f_{0:05d}'.format(i)
        i = i + 1
        if len(train_i) != 0:
            train_len = train_i.batch_size
            train_data['num_samples'].append(train_len)
            train_data['users'].append(uname)
            x = train_i.dataset.tensors[0].tolist()
            xt = []
            for xi in x:
                t = convert(xi)
                xt.append(t)
            train_data['user_data'][uname] = {'y': train_i.dataset.tensors[1].tolist(),
                                              'x': xt}
    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)

    i = 0
    test_path = 'data/MNIST/iid/test/all_data_0_niid_0_keep_10_test_9.json'
    for test_i in mnist_iid_test_dls:
        uname = 'f_{0:05d}'.format(i)
        i = i + 1
        if len(test_i) != 0:
            test_len = test_i.batch_size
            test_data['users'].append(uname)
            x = test_i.dataset.tensors[0].tolist()
            xt = []
            for xi in x:
                t = convert(xi)
                xt.append(t)
            test_data['user_data'][uname] = {'y': test_i.dataset.tensors[1].tolist(), 'x': xt}
            test_data['num_samples'].append(test_len)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
