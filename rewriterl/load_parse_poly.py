from lark import Lark
import pickle
from pathlib import Path

# from rewriterl.sympyt import record_difficulty
from lark import Transformer, Token, Tree, Visitor
from rewriterl.tree_ops import construct_integer, calculate_value

here = Path(__file__).parent


def parse_polynomial(string):

    l = Lark(
        """?start: sum
        
                        ?sum: product
                        | sum "+" product -> add
                        
                        ?product: power
                        | product "*" power -> mul
                        
                        ?power: atom
                        | power "^" atom -> pow
                        
                        ?atom: NUMBER
                        | WORD
                        | "(" sum ")"    
    
                        %import common.NUMBER
                        %import common.WORD
                        %ignore " "           // Disregard spaces in text
                     """,
        parser="lalr",
    )

    return l.parse(string)


def get_dataset(
    chunk=40,
    full_dataset=False,
    poly=1,
    set="train",
    postfix="before",
    difficulty=False,
):
    dataset = []
    filter_list = []
    with open(here / f"data/polynomial/poly{poly}/{set}.{postfix}") as f:

        if full_dataset:
            lines = f.readlines()
        else:
            lines = f.readlines()[0:chunk]
        for e, line in enumerate(lines):

            line = line.strip()
            problem = line
            # print(e, problem)

            if not difficulty:
                dataset.append([parse_polynomial(problem), e])
            elif difficulty:
                # This only works if you have a record_difficulty function, for example an instrumented sympy expand
                diff = record_difficulty(problem)

                dataset.append([parse_polynomial(problem), diff, e])

    return dataset


class MaximumNumber(Transformer):

    max_number = 0

    def NUMBER(self, tree):

        if int(tree.value) > self.max_number:
            self.max_number = int(tree.value)

        return tree


class T(Transformer):

    """
    This is a lark transformer function to replace integers by their successor representation.

    """

    def NUMBER(self, nd):

        new_int = construct_integer(int(nd.value))
        # print(new_int)

        return new_int


def transform_dataset(dataset):

    """"
    This function converts numbers (i.e. 2) to a successor tree (i.e. suc(suc(0))) so that the algorithms
    can reason over all positive integers.
    """
    fix_dataset = []
    for problem, no in dataset:
        # print(no, problem)
        if type(problem) == Tree:
            t = T(visit_tokens=True)
            fix_dataset.append([t.transform(problem), no])
        elif type(problem) == Token:
            if problem.type == "NUMBER":
                fix_dataset.append([construct_integer(int(problem.value)), no])
            else:
                fix_dataset.append([problem, no])

    return fix_dataset


def get_transformed_dataset(set="train"):

    dataset = get_dataset(chunk=40)

    return transform_dataset(dataset)


def get_transformed_dataset_goals(chunk=40, poly=1, set="train", threshold=100):

    dataset_before = get_dataset(chunk=chunk, poly=poly, set=set, postfix="before")
    dataset_after = get_dataset(chunk=chunk, poly=poly, set=set, postfix="after")

    assert len(dataset_before) == len(dataset_after)

    vis = MaximumNumber(visit_tokens=True)
    max_numbers = []
    for problem, no in dataset_after:
        vis.max_number = 0
        if type(problem) == Tree:
            vis.transform(problem)
            max_numbers.append(vis.max_number)
        elif type(problem) == Token:
            if problem.type == "NUMBER":
                max_numbers.append(int(problem.value))
            elif problem.type == "WORD":
                max_numbers.append(-1)
        else:
            raise ValueError("huh?")

    # print("Max numbers")
    # print(max_numbers)
    dataset_before = [
        k for e, k in enumerate(dataset_before) if max_numbers[e] < threshold
    ]
    dataset_after = [
        k for e, k in enumerate(dataset_after) if max_numbers[e] < threshold
    ]
    assert len(dataset_before) == len(dataset_after)
    print(
        f"After filtering with a maximum number threshold of {threshold}, {len(dataset_after)}/{chunk} problems are left."
    )
    print(len(dataset_before))
    # raise ValueError("no.")
    return list(
        zip(transform_dataset(dataset_before), transform_dataset(dataset_after))
    )


def get_transformed_dataset_difficulty(chunk=40, poly=1, set="train", threshold=100):

    dataset_before = get_dataset(
        chunk=chunk, poly=poly, set=set, postfix="before", difficulty=True
    )
    dataset_after = get_dataset(chunk=chunk, poly=poly, set=set, postfix="after")

    assert len(dataset_before) == len(dataset_after)

    vis = MaximumNumber(visit_tokens=True)
    max_numbers = []
    for problem, no in dataset_after:
        vis.max_number = 0
        if type(problem) == Tree:
            vis.transform(problem)
            max_numbers.append(vis.max_number)
        elif type(problem) == Token:
            if problem.type == "NUMBER":
                max_numbers.append(int(problem.value))
            elif problem.type == "WORD":
                max_numbers.append(-1)
        else:
            raise ValueError("huh?")

    # print("Max numbers")
    # print(max_numbers)
    dataset_before = [
        k for e, k in enumerate(dataset_before) if max_numbers[e] < threshold
    ]
    dataset_after = [
        k for e, k in enumerate(dataset_after) if max_numbers[e] < threshold
    ]
    assert len(dataset_before) == len(dataset_after)
    print(
        f"After filtering with a maximum number threshold of {threshold}, {len(dataset_after)}/{chunk} problems are left."
    )
    print(len(dataset_before))
    # raise ValueError("no.")
    diff = [k[1] for k in dataset_before]
    dataset_before = [(k[0], k[2]) for k in dataset_before]

    tr_before = transform_dataset(dataset_before)
    tr_after = transform_dataset(dataset_after)

    trees_before = [k[0] for k in tr_before]
    trees_after = [k[0] for k in tr_after]
    index = [k[1] for k in tr_before]

    return list(zip(trees_before, trees_after, diff, index))


def construct_curriculum(size, set="train"):

    """"
    size: number of problems to subsample from each poly dataset - total size of dataset will be 6x this

    """

    total_problem_list = []

    for poly in [1, 2, 3, 4, 5, 6]:
        d = get_transformed_dataset_difficulty(chunk=size, poly=poly, set=set)
        indexed_d = []
        for e, problem in enumerate(d):
            # append the original dataset, which in combination with the index will allow
            # finding the original line for the problem in the datafile
            indexed_d.append(list(problem) + [poly])
        total_problem_list += indexed_d

    total_problem_list.sort(key=lambda x: x[2], reverse=False)  # ascending difficulty

    # assign new indices
    global_index_problems = []
    for e, problem in enumerate(total_problem_list):
        global_index_problems.append(list(problem) + [e])

    # relabel the indices
    return global_index_problems


def load_curriculum():

    import pickle

    curr = pickle.load(open(here / "data/polynomial/curriculum.data", "rb"))

    return curr


def load_curriculum_test():

    import pickle

    curr = pickle.load(open(here / "data/polynomial/curriculum_test.data", "rb"))

    return curr


def remove_instant_wins(transformed_dataset):
    new_data = []
    for ((before, no1), (after, no2)) in transformed_dataset:
        assert no1 == no2
        if not before == after:
            new_data.append([before, after, no1])

    return new_data


def remove_instant_wins_curriculum(transformed_dataset):
    new_data = []
    for (before, after, diff, oi, ds, index) in transformed_dataset:
        if not before == after:
            new_data.append([before, after, diff, oi, ds, index])

    return new_data


# How to get the curriculums if you have instrumented the Sympy expand function
# curriculum = construct_curriculum(1000, set="test")
# import pickle
# pickle.dump(curriculum, open(here / "data/polynomial/curriculum_test.data", "wb"))
