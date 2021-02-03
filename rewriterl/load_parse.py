from lark import Lark
import pickle
from pathlib import Path
from rewriterl.tree_ops import get_size

here = Path(__file__).parent


def parse_robinson(string):

    l = Lark(
        """?start: expression
    
                ?expression: atom
                | "+" expression expression -> add
                | "*" expression expression -> mul
                | "SUC" expression -> suc
                | "(" expression ")"
    
                ?atom: NUMBER                
                   
                %import common.NUMBER
                %ignore " "           // Disregard spaces in text
             """,
        parser="lalr",
    )

    return l.parse(string)


def get_dataset(chunk=40, full_dataset=False):

    dataset = []
    with open(here / "data/robinson/train") as f:

        if full_dataset:
            lines = f.readlines()
        else:
            lines = f.readlines()[0:chunk]
        for e, line in enumerate(lines):

            line = line.strip()
            problem, value, difficulty = line.split(",")[0:3]
            if difficulty == "?":
                difficulty = -1
            dataset.append([parse_robinson(problem), int(value), int(difficulty), e])

    return dataset


def preload_dataset(set):

    if not set in ["train", "valid", "test"]:
        raise ValueError("Dataset not 'train', 'valid' or 'test'")

    print(f"Pre-loading dataset: {set}")
    # Open/create the pickle file
    file = open(here / "data/robinson/{}.pkl".format(set), "wb")

    dataset = []
    # Open the raw data file
    with open(here / "data/robinson/{}".format(set)) as f:

        lines = f.readlines()
        for e, line in enumerate(lines):
            line = line.strip()
            problem, value, difficulty = line.split(",")[0:3]
            if difficulty == "?":
                difficulty = -1
            dataset.append([parse_robinson(problem), int(value), int(difficulty), e])

    pickle.dump(dataset, file)


def load_dataset(set):

    if not set in ["train", "valid", "test"]:
        raise ValueError("Dataset not 'train', 'valid' or 'test'")

    print(f"Loading dataset: {set}")
    with open(here / "data/robinson/{}.pkl".format(set), "rb") as pickle_file:
        return pickle.load(pickle_file)


def define_levels_size(dataset):

    dataset.sort(key=lambda x: get_size(x[0]))
    print([get_size(k[0]) for k in dataset])
    dataset_lopl = [k for k in dataset if not k[2] == -1]

    levels = []
    index = 0
    while index + 400 < len(dataset_lopl):
        levels.append(dataset_lopl[index : index + 400])
        index += 400

    levels.append(dataset_lopl[index:])
    return levels


def define_levels(dataset, no_lopl=False):

    dataset.sort(key=lambda x: x[2])

    dataset_lopl = [k for k in dataset if not k[2] == -1]
    dataset_nolopl = [k for k in dataset if k[2] == -1]
    levels = []
    index = 0
    while index + 400 < len(dataset_lopl):
        levels.append(dataset_lopl[index : index + 400])
        index += 400

    levels.append(dataset_lopl[index:])

    if no_lopl:
        filler = 400 - len(levels[-1])
        # print(filler)
        levels[-1] += dataset_nolopl[:filler]
        index = filler
        # TODO: Refactor this code to be more clear
        # print("Index", index)
        while index + 400 < len(dataset_nolopl):
            levels.append(dataset_lopl[index : index + 400])
            index += 400

        levels.append(dataset_nolopl[index:])

    return levels


# preload_dataset("test")
