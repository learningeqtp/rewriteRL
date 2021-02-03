import pytest
from rewriterl.load_parse import get_dataset
from rewriterl.tree_ops_polynomial import (
    power_zero_l,
    recursive_power_l,
    recursive_power_r,
    associativity_l,
    associativity_r,
    distributivity__times_l,
    distributivity__times_r,
    times_identity_l,
    times_identity_r,
    power_of_one_l,
    first_power_of_x_l,
    first_power_of_x_r,
    distributivity_power_plus_l,
    distributivity_power_plus_r,
    distributivity_power_times_l,
    distributivity_power_times_r,
    distributivity_power_power_l,
    distributivity_power_power_r,
)
from rewriterl.tree_ops import label_parents
from lark import Visitor, Tree, Transformer, v_args, tree, Token


def test_power_zero_l():

    example = Tree("suc", [Tree("pow", [Token("WORD", "x"), Token("NUMBER", "0")])])
    example = label_parents(example)

    pow_op = example.children[0]

    cursor, entire = power_zero_l(pow_op, example)

    # Case of operation being not root
    # suc
    #   pow <-----
    #     x
    #     0

    # Expected result:
    # suc
    #   suc	0

    assert repr(entire) == "Tree(suc, [Tree(suc, [Token(NUMBER, '0')])])"

    example = Tree("pow", [Token("WORD", "x"), Token("NUMBER", "0")])
    example = label_parents(example)

    cursor, entire = power_zero_l(example, example)

    # # Case of operation being root
    # pow
    #   x
    #   0

    # Expected result:
    # suc	0

    assert repr(entire) == "Tree(suc, [Token(NUMBER, '0')])"


def test_recursive_power_l():
    # x ^ s(y) = x ^ y * x
    example = Tree(
        "suc", [Tree("pow", [Token("WORD", "x"), Tree("suc", [Token("NUMBER", "0")])])]
    )

    example = label_parents(example)

    pow_op = example.children[0]

    cursor, entire = recursive_power_l(pow_op, example)

    # Case of not root:
    # suc
    #   pow <----
    #     x
    #     suc	0

    # Expected result:
    # suc
    #   mul
    #       pow
    #           x
    #           0
    #   x

    assert (
        repr(entire)
        == "Tree(suc, [Tree(mul, [Tree(pow, [Token(WORD, 'x'), Token(NUMBER, '0')]), Token(WORD, 'x')])])"
    )

    example = Tree("pow", [Token("WORD", "x"), Tree("suc", [Token("NUMBER", "0")])])

    example = label_parents(example)

    pow_op = example

    cursor, entire = recursive_power_l(pow_op, example)

    # Case of root:
    #   pow <----
    #     x
    #     suc	0

    # Expected result:
    # mul
    #   pow
    #       x
    #       0
    #   x

    assert (
        repr(entire)
        == "Tree(mul, [Tree(pow, [Token(WORD, 'x'), Token(NUMBER, '0')]), Token(WORD, 'x')])"
    )


def test_recursive_power_r():
    # x ^ y * x -> x ^ s(y)

    example = Tree(
        "suc",
        [
            Tree(
                "mul",
                [
                    Tree("pow", [Token("WORD", "x"), Token("NUMBER", "0")]),
                    Token("WORD", "x"),
                ],
            )
        ],
    )
    example = label_parents(example)

    pow_op = example.children[0]

    cursor, entire = recursive_power_r(pow_op, example)

    # Case of not root
    # suc
    #   mul <----
    #     pow
    #       x
    #       0
    #     x

    # Expected result
    # suc
    #   pow
    #       x
    #       suc
    #          0

    assert (
        repr(entire)
        == "Tree(suc, [Tree(pow, [Token(WORD, 'x'), Tree(suc, [Token(NUMBER, '0')])])])"
    )

    example = Tree(
        "mul",
        [Tree("pow", [Token("WORD", "x"), Token("NUMBER", "0")]), Token("WORD", "x")],
    )

    example = label_parents(example)

    # Case of root
    #   mul <----
    #     pow
    #       x
    #       0
    #     x

    # Expected result
    # pow
    #   x
    #   suc
    #       0

    cursor, entire = recursive_power_r(example, example)

    assert (
        repr(entire) == "Tree(pow, [Token(WORD, 'x'), Tree(suc, [Token(NUMBER, '0')])])"
    )


def test_associativity_l():
    # (x + y) + z = x + (y + z)

    example = Tree(
        "suc",
        [
            Tree(
                "add",
                [
                    Tree("add", [Token("WORD", "x"), Token("WORD", "y")]),
                    Token("WORD", "z"),
                ],
            )
        ],
    )

    example = label_parents(example)
    pow_op = example.children[0]

    cursor, entire = associativity_l(pow_op, example)

    # Case of not root
    # suc
    #   add <-----
    #     add
    #       x
    #       y
    #     z

    # Expected result:
    # suc
    #   add
    #     x
    #     add
    #       y
    #       z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(add, [Token(WORD, 'x'), Tree(add, [Token(WORD, 'y'), Token(WORD, 'z')])])])"
    )

    example = Tree(
        "add",
        [Tree("add", [Token("WORD", "x"), Token("WORD", "y")]), Token("WORD", "z"),],
    )
    example = label_parents(example)

    cursor, entire = associativity_l(example, example)

    # Case of root
    # add
    #   add
    #     x
    #     y
    #   z

    # Expected result:
    # add
    #   x
    #   add
    #     y
    #     z

    assert (
        repr(entire)
        == "Tree(add, [Token(WORD, 'x'), Tree(add, [Token(WORD, 'y'), Token(WORD, 'z')])])"
    )


def test_associativity_r():

    #  x + (y + z)  =  (x + y) + z

    example = Tree(
        "suc",
        [
            Tree(
                "add",
                [
                    Token("WORD", "x"),
                    Tree("add", [Token("WORD", "y"), Token("WORD", "z")]),
                ],
            )
        ],
    )
    example = label_parents(example)

    op = example.children[0]

    cursor, entire = associativity_r(op, example)

    # Case of not root
    # suc
    #   add <-------
    #     x
    #     add
    #       y
    #       z

    # Expected result
    # suc
    #   add
    #     add
    #       x
    #       y
    #     z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(add, [Tree(add, [Token(WORD, 'x'), Token(WORD, 'y')]), Token(WORD, 'z')])])"
    )

    example = Tree(
        "add",
        [Token("WORD", "x"), Tree("add", [Token("WORD", "y"), Token("WORD", "z")]),],
    )

    example = label_parents(example)

    cursor, entire = associativity_r(example, example)

    # Case of root
    # add
    #   x
    #   add
    #     y
    #     z

    # Expected result
    # add
    #   add
    #     x
    #     y
    #   z

    assert (
        repr(entire)
        == "Tree(add, [Tree(add, [Token(WORD, 'x'), Token(WORD, 'y')]), Token(WORD, 'z')])"
    )


def test_distributivity_times_l():

    # x * (y + z) = x * y + x * z
    #

    example = Tree(
        "suc",
        [
            Tree(
                "mul",
                [
                    Token("WORD", "x"),
                    Tree("add", [Token("WORD", "y"), Token("WORD", "z")]),
                ],
            )
        ],
    )
    example = label_parents(example)

    op = example.children[0]

    cursor, entire = distributivity__times_l(op, example)

    # Case of not root:
    # suc
    #   mul <--------
    #     x
    #     add
    #       y
    #       z

    # Expected result:
    # suc
    #   add
    #     mul
    #       x
    #       y
    #     mul
    #       x
    #       z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(add, [Tree(mul, [Token(WORD, 'x'), Token(WORD, 'y')]), Tree(mul, [Token(WORD, 'x'), Token(WORD, 'z')])])])"
    )

    example = Tree(
        "mul",
        [Token("WORD", "x"), Tree("add", [Token("WORD", "y"), Token("WORD", "z")]),],
    )

    example = label_parents(example)

    cursor, entire = distributivity__times_l(example, example)

    # Case of root
    # mul
    #   x
    #   add
    #     y
    #     z

    # Expected result
    # add
    #   mul
    #     x
    #     y
    #   mul
    #     x
    #     z

    assert (
        repr(entire)
        == "Tree(add, [Tree(mul, [Token(WORD, 'x'), Token(WORD, 'y')]), Tree(mul, [Token(WORD, 'x'), Token(WORD, 'z')])])"
    )


def test_distributivity_times_r():
    #   x * y + x * z  = x * (y + z)

    example = Tree(
        "suc",
        [
            Tree(
                "add",
                [
                    Tree("mul", [Token("WORD", "x"), Token("WORD", "y")]),
                    Tree("mul", [Token("WORD", "x"), Token("WORD", "z")]),
                ],
            )
        ],
    )
    example = label_parents(example)

    op = example.children[0]

    cursor, entire = distributivity__times_r(op, example)

    # Case of not root
    # suc
    #   add
    #     mul
    #       x
    #       y
    #     mul
    #       x
    #       z

    # Expected result:
    # suc
    #   mul
    #     x
    #     add
    #       y
    #       z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(mul, [Token(WORD, 'x'), Tree(add, [Token(WORD, 'y'), Token(WORD, 'z')])])])"
    )

    example = Tree(
        "add",
        [
            Tree("mul", [Token("WORD", "x"), Token("WORD", "y")]),
            Tree("mul", [Token("WORD", "x"), Token("WORD", "z")]),
        ],
    )

    example = label_parents(example)
    cursor, entire = distributivity__times_r(example, example)

    # Case of root
    # add
    #   mul
    #     x
    #     y
    #   mul
    #     x
    #     z

    # Expected result
    # mul
    #   x
    #   add
    #     y
    #     z

    assert (
        repr(entire)
        == "Tree(mul, [Token(WORD, 'x'), Tree(add, [Token(WORD, 'y'), Token(WORD, 'z')])])"
    )


def test_times_identity_l():

    # x * 1 = x
    example = Tree(
        "suc", [Tree("mul", [Token("WORD", "x"), Tree("suc", [Token("NUMBER", "0")])])]
    )
    example = label_parents(example)
    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = times_identity_l(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc
    #   mul <----
    #     x
    #     suc	0

    # Expected result
    # suc	x

    assert repr(entire) == "Tree(suc, [Token(WORD, 'x')])"

    example = Tree("mul", [Token("WORD", "x"), Tree("suc", [Token("NUMBER", "0")])])

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = times_identity_l(example, example)

    print("b")
    print(entire)
    print(repr(entire))

    # case of root
    # mul
    #   x
    #   suc	0

    # Expected result
    # x

    assert repr(entire) == "Token(WORD, 'x')"


def test_times_identity_r():

    # x = x * 1

    example = Tree("suc", [Token("WORD", "x")])
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = times_identity_r(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc	x

    # Expected result
    # suc
    #   mul
    #     x
    #     suc	0

    assert (
        repr(entire)
        == "Tree(suc, [Tree(mul, [Token(WORD, 'x'), Tree(suc, [Token(NUMBER, '0')])])])"
    )

    example = Token("WORD", "x")
    example = label_parents(example)

    print("b")
    print(example)

    cursor, entire = times_identity_r(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of root
    # x

    # Expected result
    # mul
    #   x
    #   suc	0

    assert (
        repr(entire) == "Tree(mul, [Token(WORD, 'x'), Tree(suc, [Token(NUMBER, '0')])])"
    )


def test_power_of_one_l():

    # 1 ^ x = 1

    example = Tree(
        "suc", [Tree("pow", [Tree("suc", [Token("NUMBER", "0")]), Token("WORD", "x")])]
    )
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = power_of_one_l(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc
    #   pow
    #     suc	0
    #     x

    # Expected result
    # suc
    #   suc	0

    assert repr(entire) == "Tree(suc, [Tree(suc, [Token(NUMBER, '0')])])"

    example = Tree("pow", [Tree("suc", [Token("NUMBER", "0")]), Token("WORD", "x")])

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = power_of_one_l(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of root
    # pow
    #   suc	0
    #   x

    # Expected result
    # suc	0

    assert repr(entire) == "Tree(suc, [Token(NUMBER, '0')])"


def test_first_power_of_x_l():

    # x ^ 1 = x

    example = Tree(
        "suc", [Tree("pow", [Token("WORD", "x"), Tree("suc", [Token("NUMBER", "0")])])]
    )

    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = first_power_of_x_l(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc
    #   pow
    #     x
    #     suc	0

    # Expected results
    # suc	x

    assert repr(entire) == "Tree(suc, [Token(WORD, 'x')])"

    example = Tree("pow", [Token("WORD", "x"), Tree("suc", [Token("NUMBER", "0")])])

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = first_power_of_x_l(example, example)

    print("b")
    print(entire)
    print(repr(entire))

    # Case of root
    # pow
    #   x
    #   suc	0

    # Expected result
    # x

    assert repr(entire) == "Token(WORD, 'x')"


def test_first_power_of_x_r():

    # x = x ^ 1

    example = Tree("suc", [Token("WORD", "x")])
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = first_power_of_x_r(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc	x

    # Expected result
    # suc
    #   pow
    #     x
    #     suc	0

    assert (
        repr(entire)
        == "Tree(suc, [Tree(pow, [Token(WORD, 'x'), Tree(suc, [Token(NUMBER, '0')])])])"
    )

    example = Token("WORD", "x")
    example = label_parents(example)

    print("b")
    print(example)

    cursor, entire = first_power_of_x_r(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of root:
    # x

    # Expected result
    # pow
    #   x
    #   suc	0

    assert (
        repr(entire) == "Tree(pow, [Token(WORD, 'x'), Tree(suc, [Token(NUMBER, '0')])])"
    )


def test_distributivity_power_plus_l():

    # x ^ (y + z) = x ^ y * x ^ z

    example = Tree(
        "suc",
        [
            Tree(
                "pow",
                [
                    Token("WORD", "x"),
                    Tree("add", [Token("WORD", "y"), Token("WORD", "z")]),
                ],
            )
        ],
    )
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = distributivity_power_plus_l(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc
    #   pow <----
    #     x
    #     add
    #       y
    #       z

    # Expected result
    # suc
    #   mul
    #     pow
    #       x
    #       y
    #     pow
    #       x
    #       z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(mul, [Tree(pow, [Token(WORD, 'x'), Token(WORD, 'y')]), Tree(pow, [Token(WORD, 'x'), Token(WORD, 'z')])])])"
    )

    example = Tree(
        "pow",
        [Token("WORD", "x"), Tree("add", [Token("WORD", "y"), Token("WORD", "z")]),],
    )

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = distributivity_power_plus_l(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of root
    # pow
    #   x
    #   add
    #     y
    #     z

    # mul
    #   pow
    #     x
    #     y
    #   pow
    #     x
    #     z

    assert (
        repr(entire)
        == "Tree(mul, [Tree(pow, [Token(WORD, 'x'), Token(WORD, 'y')]), Tree(pow, [Token(WORD, 'x'), Token(WORD, 'z')])])"
    )


def test_distributivity_power_plus_r():
    # x ^ y * x ^ z = x ^ (y + z)

    example = Tree(
        "suc",
        [
            Tree(
                "mul",
                [
                    Tree("pow", [Token("WORD", "x"), Token("WORD", "y")]),
                    Tree("pow", [Token("WORD", "x"), Token("WORD", "z")]),
                ],
            )
        ],
    )
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = distributivity_power_plus_r(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc
    #   mul <-----
    #     pow
    #       x
    #       y
    #     pow
    #       x
    #       z

    # Expected result
    # suc
    #   pow
    #     x
    #     add
    #       y
    #       z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(pow, [Token(WORD, 'x'), Tree(add, [Token(WORD, 'y'), Token(WORD, 'z')])])])"
    )

    example = Tree(
        "mul",
        [
            Tree("pow", [Token("WORD", "x"), Token("WORD", "y")]),
            Tree("pow", [Token("WORD", "x"), Token("WORD", "z")]),
        ],
    )

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = distributivity_power_plus_r(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # case of root
    # mul
    #   pow
    #     x
    #     y
    #   pow
    #     x
    #     z

    # Expected result
    # pow
    #   x
    #   add
    #     y
    #     z

    assert (
        repr(entire)
        == "Tree(pow, [Token(WORD, 'x'), Tree(add, [Token(WORD, 'y'), Token(WORD, 'z')])])"
    )


def test_distributivity_power_times_l():

    # (x * y) ^ z = x ^ z * y ^ z

    example = Tree(
        "suc",
        [
            Tree(
                "pow",
                [
                    Tree("mul", [Token("WORD", "x"), Token("WORD", "y")]),
                    Token("WORD", "z"),
                ],
            )
        ],
    )
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = distributivity_power_times_l(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of  not root
    # suc
    #   pow <----
    #     mul
    #       x
    #       y
    #     z

    # Expected result
    # suc
    #   mul
    #     pow
    #       x
    #       z
    #     pow
    #       y
    #       z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(mul, [Tree(pow, [Token(WORD, 'x'), Token(WORD, 'z')]), Tree(pow, [Token(WORD, 'y'), Token(WORD, 'z')])])])"
    )

    example = Tree(
        "pow",
        [Tree("mul", [Token("WORD", "x"), Token("WORD", "y")]), Token("WORD", "z"),],
    )

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = distributivity_power_times_l(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of root
    # pow
    #   mul
    #     x
    #     y
    #   z

    # Expected result
    # mul
    #   pow
    #     x
    #     z
    #   pow
    #     y
    #     z

    assert (
        repr(entire)
        == "Tree(mul, [Tree(pow, [Token(WORD, 'x'), Token(WORD, 'z')]), Tree(pow, [Token(WORD, 'y'), Token(WORD, 'z')])])"
    )


def test_distributivity_power_times_r():

    #   x ^ z * y ^ z = (x * y) ^ z

    example = Tree(
        "suc",
        [
            Tree(
                "mul",
                [
                    Tree("pow", [Token("WORD", "x"), Token("WORD", "z")]),
                    Tree("pow", [Token("WORD", "y"), Token("WORD", "z")]),
                ],
            )
        ],
    )
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = distributivity_power_times_r(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc
    #   mul <-----
    #     pow
    #       x
    #       z
    #     pow
    #       y
    #       z

    # Expected result
    # suc
    #   pow
    #     mul
    #       x
    #       y
    #     z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(pow, [Tree(mul, [Token(WORD, 'x'), Token(WORD, 'y')]), Token(WORD, 'z')])])"
    )

    example = Tree(
        "mul",
        [
            Tree("pow", [Token("WORD", "x"), Token("WORD", "z")]),
            Tree("pow", [Token("WORD", "y"), Token("WORD", "z")]),
        ],
    )

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = distributivity_power_times_r(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of root
    # mul
    #   pow
    #     x
    #     z
    #   pow
    #     y
    #     z

    # Expected result
    # pow
    #   mul
    #     x
    #     y
    #   z

    assert (
        repr(entire)
        == "Tree(pow, [Tree(mul, [Token(WORD, 'x'), Token(WORD, 'y')]), Token(WORD, 'z')])"
    )


def test_distributivity_power_power_l():

    # (x ^ y) ^ z= x ^ (y * z)

    example = Tree(
        "suc",
        [
            Tree(
                "pow",
                [
                    Tree("pow", [Token("WORD", "x"), Token("WORD", "y")]),
                    Token("WORD", "z"),
                ],
            )
        ],
    )
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = distributivity_power_power_l(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc
    #   pow <-----
    #     pow
    #       x
    #       y
    #     z

    # Expected result
    # suc
    #   pow
    #     x
    #     mul
    #       y
    #       z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(pow, [Token(WORD, 'x'), Tree(mul, [Token(WORD, 'y'), Token(WORD, 'z')])])])"
    )

    example = Tree(
        "pow",
        [Tree("pow", [Token("WORD", "x"), Token("WORD", "y")]), Token("WORD", "z"),],
    )

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = distributivity_power_power_l(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of root
    # pow
    #   pow
    #     x
    #     y
    #   z

    # Expected result
    # pow
    #   x
    #   mul
    #     y
    #     z

    assert (
        repr(entire)
        == "Tree(pow, [Token(WORD, 'x'), Tree(mul, [Token(WORD, 'y'), Token(WORD, 'z')])])"
    )


def test_distributivity_power_power_r():

    # x ^ (y * z)  = (x ^ y) ^ z

    example = Tree(
        "suc",
        [
            Tree(
                "pow",
                [
                    Token("WORD", "x"),
                    Tree("mul", [Token("WORD", "y"), Token("WORD", "z")]),
                ],
            )
        ],
    )
    example = label_parents(example)

    print("b")
    print(example.pretty())

    op = example.children[0]

    cursor, entire = distributivity_power_power_r(op, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of not root
    # suc
    #   pow
    #     x
    #     mul
    #       y
    #       z

    # Expected result
    # suc
    #   pow
    #     pow
    #       x
    #       y
    #     z

    assert (
        repr(entire)
        == "Tree(suc, [Tree(pow, [Tree(pow, [Token(WORD, 'x'), Token(WORD, 'y')]), Token(WORD, 'z')])])"
    )

    example = Tree(
        "pow",
        [Token("WORD", "x"), Tree("mul", [Token("WORD", "y"), Token("WORD", "z")]),],
    )

    example = label_parents(example)

    print("b")
    print(example.pretty())

    cursor, entire = distributivity_power_power_r(example, example)

    print("b")
    print(entire.pretty())
    print(repr(entire))

    # Case of root
    # pow
    #   x
    #   mul
    #     y
    #     z

    # Expected result
    # pow
    #   pow
    #     x
    #     y
    #   z

    assert (
        repr(entire)
        == "Tree(pow, [Tree(pow, [Token(WORD, 'x'), Token(WORD, 'y')]), Token(WORD, 'z')])"
    )


# test_power_zero_l()
