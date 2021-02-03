import pytest
from rewriterl.load_parse import get_dataset
from rewriterl.tree_ops import (
    winning_state,
    check_tree,
    label_parents,
    plus_zero_l,
    plus_zero_r,
    recursive_plus_l,
    recursive_plus_r,
    times_zero_l,
    recursive_times_l,
    recursive_times_r,
    calculate_value,
)

# TODO Include the pretty prints for the expected answers everywhere


def test_calculate_value():

    dataset = get_dataset(chunk=10)
    l = dataset[2][0]
    l = label_parents(l)
    winning = winning_state(l)

    assert calculate_value(l) == 2


def test_winning_state():

    # Should be true
    dataset = get_dataset(chunk=130)
    l = dataset[120][0]
    l = label_parents(l)

    winning = winning_state(l)

    assert winning

    # Should be true
    dataset = get_dataset(chunk=130)
    l = dataset[121][0]
    l = label_parents(l)

    winning = winning_state(l)

    assert not winning


def test_check_tree():

    # All beginning states should be well formed
    dataset = get_dataset(chunk=130)
    for i, formula in enumerate(dataset):

        formula = label_parents(formula)
        wellformed = check_tree(formula)

        assert wellformed


def test_plus_zero_l():
    dataset = get_dataset(chunk=60)

    # Case of add operation being not root
    # mul
    #   0
    #   add
    #     0
    #     0
    l = dataset[51][0]
    l = label_parents(l)
    add_op = l.children[1]  # move to add op
    add_op, l = plus_zero_l(add_op, l)

    assert repr(add_op) == "Tree(mul, [Token(NUMBER, '0'), Token(NUMBER, '0')])"

    # Case of add operation being root
    # add
    #   0
    #   0

    l = dataset[8][0]
    l = label_parents(l)
    add_op = l
    add_op, l = plus_zero_l(add_op, l)

    assert repr(l) == "Token(NUMBER, '0')"


def test_plus_zero_r():

    dataset = get_dataset(chunk=60)
    # Case of not root
    # suc
    #   mul <--- cursor here
    #     0
    #     0
    l = dataset[13][0]
    l = label_parents(l)
    root = l.children[0]
    add_op, l = plus_zero_r(root, l)

    # Expected result
    # suc
    #   add
    #     mul
    #       0
    #       0
    #     0

    assert (
        repr(l)
        == "Tree(suc, [Tree(add, [Tree(mul, [Token(NUMBER, '0'), Token(NUMBER, '0')]), Token(NUMBER, '0')])])"
    )
    # Case of root
    # suc	0
    l = dataset[1][0]
    l = label_parents(l)

    add_op, l = plus_zero_r(l, l)

    # Expected output
    # add
    #   suc	0
    #   0

    assert repr(l) == "Tree(add, [Tree(suc, [Token(NUMBER, '0')]), Token(NUMBER, '0')])"


def test_recursive_plus_l():
    dataset = get_dataset(chunk=60)

    # Case where add is not root
    # suc
    #   add
    #     suc	0
    #     suc	0
    l = dataset[20][0]
    l = label_parents(l)
    tree = l.children[0]
    add_op, l = recursive_plus_l(tree, l)

    assert (
        repr(l)
        == "Tree(suc, [Tree(suc, [Tree(add, [Tree(suc, [Token(NUMBER, '0')]), Token(NUMBER, '0')])])])"
    )

    # Case where add is root
    # add
    #   0
    #   suc	0
    l = dataset[9][0]
    l = label_parents(l)
    add_op, l = recursive_plus_l(l, l)
    assert repr(l) == "Tree(suc, [Tree(add, [Token(NUMBER, '0'), Token(NUMBER, '0')])])"


def test_recursive_plus_r():
    # Case of not root
    # suc
    #   suc <-- cursor will be here
    #     add
    #       0
    #       suc	0
    dataset = get_dataset(chunk=60)

    l = dataset[35][0]
    l = label_parents(l)
    root = l.children[0]

    add_op, l = recursive_plus_r(root, l)
    # suc
    #   add
    #     0
    #     suc
    #       suc	0

    assert (
        repr(l)
        == "Tree(suc, [Tree(add, [Token(NUMBER, '0'), Tree(suc, [Tree(suc, [Token(NUMBER, '0')])])])])"
    )
    # Case of root
    # suc
    #   add
    #     0
    #     0
    dataset = get_dataset(chunk=60)
    l = dataset[17][0]
    l = label_parents(l)
    add_op, l = recursive_plus_r(l, l)

    # Expected result
    # add
    #   0
    #   suc	0
    assert repr(l) == "Tree(add, [Token(NUMBER, '0'), Tree(suc, [Token(NUMBER, '0')])])"


def test_times_zero_l():
    dataset = get_dataset(chunk=60)
    # Case where times is not root
    # mul
    #   0
    #   mul  < ---- this is where the cursor will be
    #     suc	0
    #     0
    l = dataset[49][0]
    l = label_parents(l)
    cursor = l.children[1]
    op, l = times_zero_l(cursor, l)

    assert repr(l) == "Tree(mul, [Token(NUMBER, '0'), Token(NUMBER, '0')])"
    # Result:
    # mul
    #   0
    #   0

    # Case where times is root
    # mul
    #   0
    #   0
    l = dataset[4][0]
    l = label_parents(l)

    op, l = times_zero_l(l, l)

    assert repr(l) == "Token(NUMBER, '0')"


def test_recursive_times_l():
    dataset = get_dataset(chunk=60)
    # Case where mul is not root
    # suc
    #   suc
    #     mul <--- Cursor will be here
    #       0
    #       suc	0
    l = dataset[31][0]
    l = label_parents(l)

    cursor = l.children[0].children[0]
    op, l = recursive_times_l(cursor, l)

    # Expected result
    # suc
    #   suc
    #     add
    #       mul
    #         0
    #         0
    #       0

    assert (
        repr(l)
        == "Tree(suc, [Tree(suc, [Tree(add, [Tree(mul, [Token(NUMBER, '0'), Token(NUMBER, '0')]), Token(NUMBER, '0')])])])"
    )
    # Case where mul is root
    # mul
    #   0
    #   suc	0
    l = dataset[5][0]
    l = label_parents(l)
    op, l = recursive_times_l(l, l)

    # Expected result
    # add
    #   mul
    #     0
    #     0
    #   0
    assert (
        repr(l)
        == "Tree(add, [Tree(mul, [Token(NUMBER, '0'), Token(NUMBER, '0')]), Token(NUMBER, '0')])"
    )


def test_recursive_times_r():
    dataset = get_dataset(chunk=190)

    # Case of not root
    # suc
    #   add
    #     mul
    #       suc	0
    #       suc	0
    #     suc	0

    l = dataset[180][0]
    l = label_parents(l)
    root = l.children[0]
    add_op, l = recursive_times_r(root, l)

    # expected result
    # suc
    #   mul
    #     suc	0
    #     suc
    #       suc	0

    assert (
        repr(l)
        == "Tree(suc, [Tree(mul, [Tree(suc, [Token(NUMBER, '0')]), Tree(suc, [Tree(suc, [Token(NUMBER, '0')])])])])"
    )

    # Case of root
    # add
    #   mul
    #     suc	0
    #     0
    #   suc	0

    l = dataset[109][0]
    l = label_parents(l)
    add_op, l = recursive_times_r(l, l)
    # Expected result
    # mul
    #   suc	0
    #   suc	0

    assert (
        repr(l)
        == "Tree(mul, [Tree(suc, [Token(NUMBER, '0')]), Tree(suc, [Token(NUMBER, '0')])])"
    )
