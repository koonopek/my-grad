import matplotlib.pyplot as plt
import torch
names_file = open("./data/names.txt", "r")
G = torch.Generator().manual_seed(42)
names_all = names_file.read().split("\n")
names_train, names_dev, names_test = torch.utils.data.random_split(
    names_all, [0.8, 0.1, 0.1], generator=G)


for name in names_train:
    print(name)
