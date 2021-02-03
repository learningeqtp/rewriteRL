from lark import Visitor, Tree, Transformer, v_args, tree
from lark import Token
import copy

# List of actions

# 0. x + 0 → x
# 1. x → x + 0
# 2. x + s(y) → s(x + y)
# 3. s(x + y) → x + s(y)
# 4. x * 0 → 0
# 5. x * s(y) → x * y + x
# 6. x * y + x → x * s(y)
# 7. Move cursor to left child
# 8. Move cursor to right child
# 9. x + y = y + x (commutativity) (also mul)
# 10. x ^ 0 = 1
# 11. x ^ s(y) = x ^ y * x
# 12. x ^ y * x -> x ^ s(y)
# 13. (x + y) + z = x + (y + z) (associativity) (also mul)
# 14  x + (y + z) = (x + y) + z (associativity) (also mul)
# 15. x * (y + z) = x * y + x * z
# 16. x * y + x * z = x * (y + z)
# 17. x * 1 = x
# 18. x = x * 1
# 19. 1 ^ x = 1
# 20. x ^ 1 = x
# 21. x = x ^ 1
# 22. x ^ (y + z) = x ^ y * x ^ z
# 23. x ^ y * x ^ z = x ^ (y + z)
# 24. (x * y) ^ z = x ^ z * x ^ z
# 25. x ^ z * x ^ z = (x * y) ^ z
# 26. (x ^ y) ^ z= x ^ (y * z)
# 27. x ^ (y * z)  = (x ^ y) ^ z


def poly_legal_actions(cursor):

    legal = [0] * 28

    # 0 x + 0 → x

    if hasattr(cursor, "data"):
        if cursor.data == "add" and repr(cursor.children[1]) == "Token(NUMBER, '0')":
            legal[0] = 1
    # 1 x → x + 0
    # Can always add zero

    legal[1] = 1

    # 2  x + s(y) → s(x + y)
    if hasattr(cursor, "data"):
        if cursor.data == "add":
            if hasattr(cursor.children[1], "data"):

                if cursor.children[1].data == "suc":
                    legal[2] = 1

    # 3 s(x + y) → x + s(y)
    if hasattr(cursor, "data"):
        if cursor.data == "suc":
            if hasattr(cursor.children[0], "data"):
                if cursor.children[0].data == "add":
                    legal[3] = 1

    # 4 x * 0 → 0
    if hasattr(cursor, "data"):
        if cursor.data == "mul" and repr(cursor.children[1]) == "Token(NUMBER, '0')":
            legal[4] = 1

    # 5 x * s(y) → x * y + x
    if hasattr(cursor, "data"):
        if cursor.data == "mul":
            if hasattr(cursor.children[1], "data"):
                if cursor.children[1].data == "suc":
                    legal[5] = 1

    # 6 x * y + x → x * s(y)
    if hasattr(cursor, "data"):
        if cursor.data == "add":
            if type(cursor.children[0]) == Tree:
                if cursor.children[0].data == "mul":
                    if cursor.children[0].children[0] == cursor.children[1]:
                        legal[6] = 1

    # 7 Move cursor to left child

    if hasattr(cursor, "data"):
        # Changed it here to be able to select variables
        # if type(cursor.children[0]) == Tree:
        legal[7] = 1

    # 8 Move cursor to right child
    if hasattr(cursor, "data"):
        if len(cursor.children) > 1:
            legal[8] = 1

    # 9 Commutativity of + and *
    if hasattr(cursor, "data"):

        if cursor.data == "add" or cursor.data == "mul":
            legal[9] = 1

    # 10 x ^ 0 = 1
    if hasattr(cursor, "data"):

        if cursor.data == "pow":
            if repr(cursor.children[1]) == "Token(NUMBER, '0')":
                legal[10] = 1

    # 11 x ^ s(y) = x ^ y * x

    if hasattr(cursor, "data"):
        if cursor.data == "pow":
            if type(cursor.children[1]) == Tree:
                if cursor.children[1].data == "suc":
                    legal[11] = 1

    # 12 x ^ y * x -> x ^ s(y)
    if hasattr(cursor, "data"):

        if cursor.data == "mul":
            if type(cursor.children[0]) == Tree:
                if cursor.children[0].data == "pow":
                    if cursor.children[0].children[0] == cursor.children[1]:
                        legal[12] = 1

    # 13 (x + y) + z = x + (y + z) (also mul)
    # Two separate cases, add and mul, cannot mix
    if hasattr(cursor, "data"):
        if cursor.data == "add":
            if type(cursor.children[0]) == Tree:
                if cursor.children[0].data == "add":
                    legal[13] = 1

    if hasattr(cursor, "data"):
        if cursor.data == "mul":
            if type(cursor.children[0]) == Tree:
                if cursor.children[0].data == "mul":
                    legal[13] = 1

    # 14 x + (y + z) = (x + y) + z (also mul!)
    if hasattr(cursor, "data"):
        if cursor.data == "add":
            if type(cursor.children[1]) == Tree:
                if cursor.children[1].data == "add":
                    legal[14] = 1

    if hasattr(cursor, "data"):
        if cursor.data == "mul":
            if type(cursor.children[1]) == Tree:
                if cursor.children[1].data == "mul":
                    legal[14] = 1

    # 15 x * (y + z) = x * y + x * z
    if hasattr(cursor, "data"):
        if cursor.data == "mul":
            if type(cursor.children[1]) == Tree:
                if cursor.children[1].data == "add":
                    legal[15] = 1

    # 16 x * y + x * z = x * (y + z)
    if hasattr(cursor, "data"):
        if cursor.data == "add":
            if type(cursor.children[0]) == Tree and type(cursor.children[1]) == Tree:
                if (
                    cursor.children[0].data == "mul"
                    and cursor.children[1].data == "mul"
                ):
                    if cursor.children[0].children[0] == cursor.children[1].children[0]:
                        legal[16] = 1

    # 17  x * 1 = x

    if hasattr(cursor, "data"):
        if cursor.data == "mul":
            if repr(cursor.children[1]) == "Tree(suc, [Token(NUMBER, '0')])":
                legal[17] = 1

    # 18 x = x * 1
    # Always legal
    legal[18] = 1

    # 19 1 ^ x = 1
    if hasattr(cursor, "data"):
        if cursor.data == "pow":
            if repr(cursor.children[0]) == "Tree(suc, [Token(NUMBER, '0')])":
                legal[19] = 1

    # 20 x ^ 1 = x
    if hasattr(cursor, "data"):
        if cursor.data == "pow":
            if repr(cursor.children[1]) == "Tree(suc, [Token(NUMBER, '0')])":
                legal[20] = 1

    # 21 x = x ^ 1
    # Always legal

    legal[21] = 1

    # 22 x ^ (y + z) = x ^ y * x ^ z
    if hasattr(cursor, "data"):
        if cursor.data == "pow":
            if type(cursor.children[1]) == Tree:
                if cursor.children[1].data == "add":
                    legal[22] = 1

    # 23 x ^ y * x ^ z = x ^ (y + z)
    if hasattr(cursor, "data"):
        if cursor.data == "mul":
            if type(cursor.children[0]) == Tree and type(cursor.children[1]) == Tree:
                if (
                    cursor.children[0].data == "pow"
                    and cursor.children[1].data == "pow"
                ):
                    # print("TRIGGERED 23A")
                    if cursor.children[0].children[0] == cursor.children[1].children[0]:
                        # print("TRIGGERED 23B")
                        legal[23] = 1

    # 24 (x * y) ^ z = x ^ z * y ^ z
    if hasattr(cursor, "data"):
        if cursor.data == "pow":
            if type(cursor.children[0]) == Tree:
                if cursor.children[0].data == "mul":
                    legal[24] = 1

    # 25 x ^ z * y ^ z = (x * y) ^ z
    if hasattr(cursor, "data"):
        if cursor.data == "mul":
            if type(cursor.children[0]) == Tree and type(cursor.children[1]) == Tree:
                if (
                    cursor.children[0].data == "pow"
                    and cursor.children[1].data == "pow"
                ):
                    if cursor.children[0].children[1] == cursor.children[1].children[1]:
                        legal[25] = 1

    # 26 (x ^ y) ^ z= x ^ (y * z)
    if hasattr(cursor, "data"):
        if cursor.data == "pow":
            if type(cursor.children[0]) == Tree:
                if cursor.children[0].data == "pow":
                    legal[26] = 1

    # 27 x ^ (y * z)  = (x ^ y) ^ z
    if hasattr(cursor, "data"):
        if cursor.data == "pow":
            if type(cursor.children[1]) == Tree:
                if cursor.children[1].data == "mul":
                    legal[27] = 1

    return legal


def poly_legal_actions_indices(cursor):

    legal = poly_legal_actions(cursor)
    indices = [k for k in range(28) if legal[k] == 1]
    return indices


# From lark package
def tree_poly_png(cursor, tree, filename, rankdir="LR", **kwargs):
    """Creates a colorful image that represents the tree (data+children, without meta)
    Possible values for `rankdir` are "TB", "LR", "BT", "RL", corresponding to
    directed graphs drawn from top to bottom, from left to right, from bottom to
    top, and from right to left, respectively.
    `kwargs` can be any graph attribute (e. g. `dpi=200`). For a list of
    possible attributes, see https://www.graphviz.org/doc/info/attrs.html.
    """

    import pydot

    graph = pydot.Dot(graph_type="digraph", rankdir=rankdir, **kwargs)

    i = [0]

    def new_leaf(leaf):
        node = pydot.Node(i[0], label=leaf.value)
        i[0] += 1
        graph.add_node(node)
        return node

    def _to_pydot(subtree, cursor):
        if hasattr(subtree, "data"):
            color = hash(subtree.data) & 0xFFFFFF
            color |= 0x808080

        if hasattr(subtree, "children"):
            subnodes = [
                _to_pydot(child, cursor) if isinstance(child, Tree) else new_leaf(child)
                for child in subtree.children
            ]

        if subtree is cursor:
            if type(subtree) == Tree:

                node = pydot.Node(
                    i[0], style="filled", fillcolor="red", label=subtree.data
                )
            elif type(subtree) == Token:
                node = pydot.Node(
                    i[0], style="filled", fillcolor="red", label=subtree.value
                )
        elif subtree is not cursor:
            if hasattr(subtree, "data"):
                if subtree.data == "add":
                    node = pydot.Node(
                        i[0], style="filled", fillcolor="green", label=subtree.data
                    )
                elif subtree.data == "mul":
                    node = pydot.Node(
                        i[0], style="filled", fillcolor="blue", label=subtree.data
                    )
                elif subtree.data == "suc":
                    node = pydot.Node(
                        i[0], style="filled", fillcolor="cyan", label=subtree.data
                    )
                elif subtree.data == "pow":
                    node = pydot.Node(
                        i[0], style="filled", fillcolor="orange", label=subtree.data
                    )
            else:

                node = pydot.Node(
                    i[0], style="filled", fillcolor="white", label=subtree.value
                )
        # else:
        #     node = pydot.Node(
        #         i[0], style="filled", fillcolor="#%x" % color, label=subtree.data
        #     )
        i[0] += 1
        graph.add_node(node)

        if type(subtree) == Tree:
            for subnode in subnodes:
                graph.add_edge(pydot.Edge(node, subnode))

        return node

    _to_pydot(tree, cursor)
    graph.write_png(filename)


def power_zero_l(tree, entire):

    """"Performs x ^ 0 = 1
    Give the ^ as cursor
    """
    one = Tree("suc", [Token("NUMBER", "0")])

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent
        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                grandparent.children[e] = one

        return grandparent, entire

    else:

        entire = one

        return tree, entire


def recursive_power_l(tree, entire):
    """" Performs x ^ s(y) = x ^ y * x

    Supply the cursor at *

    """
    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent
        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        # Make the new mul node
        mul = Tree("mul", [])

        grandparent.children[treeindex] = mul
        # Copy the x term that will be in both power and mul op
        x = copy.deepcopy(tree.children[0])

        # Move the child over the S() to the y
        tree.children[1] = tree.children[1].children[0]

        # Give the mul node its children
        mul.children = [tree, x]

        return grandparent, entire

    else:
        # Need a new op, the mul
        mul = Tree("mul", [])

        x = copy.deepcopy(entire.children[0])
        entire.children[1] = entire.children[1].children[0]
        mul.children = [entire, x]
        entire = mul

        return tree, entire


def recursive_power_r(tree, entire):
    """"
    Performs
    x ^ y * x -> x ^ s(y)
    """

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        new_suc = Tree("suc", [])
        # cop = tree

        new_root = tree.children[0]
        new_suc.children = [new_root.children[1]]
        new_root.children[1] = new_suc

        grandparent.children[treeindex] = new_root

        return tree, entire

    else:

        new_suc = Tree("suc", [])
        # cop = entire
        new_root = entire.children[0]
        y = new_root.children[1]
        new_suc.children = [y]
        new_root.children[1] = new_suc

        entire = new_root
        return tree, entire


def associativity_l(tree, entire):

    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0].children[0]
        y = tree.children[0].children[1]
        z = tree.children[1]

        op = tree.children[0]

        tree.children[0] = x
        op.children = [y, z]
        tree.children[1] = op

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0].children[0]
        y = entire.children[0].children[1]
        z = entire.children[1]

        op = entire.children[0]

        entire.children[0] = x
        op.children = [y, z]
        entire.children[1] = op

        return tree, entire


def associativity_r(tree, entire):
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0]
        y = tree.children[1].children[0]
        z = tree.children[1].children[1]

        op = tree.children[1]

        tree.children[1] = z
        op.children = [x, y]
        tree.children[0] = op

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0]
        y = entire.children[1].children[0]
        z = entire.children[1].children[1]

        op = entire.children[1]

        entire.children[1] = z
        op.children = [x, y]
        entire.children[0] = op

        return tree, entire


def distributivity__times_l(tree, entire):

    """"
    Performs x * (y + z) = x * y + x * z
    """
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0]
        y = tree.children[1].children[0]
        z = tree.children[1].children[1]
        x2 = copy.deepcopy(x)

        op1 = Tree("mul", [])
        op2 = Tree("mul", [])
        op3 = Tree("add", [])

        op1.children = [x, y]
        op2.children = [x2, z]
        op3.children = [op1, op2]

        tree = op3

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0]
        y = entire.children[1].children[0]
        z = entire.children[1].children[1]
        x2 = copy.deepcopy(x)

        op1 = Tree("mul", [])
        op2 = Tree("mul", [])
        op3 = Tree("add", [])

        op1.children = [x, y]
        op2.children = [x2, z]
        op3.children = [op1, op2]

        entire = op3

        return tree, entire


def distributivity__times_r(tree, entire):
    """"
    Performs  x * y + x * z = x * (y + z)
    """
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0].children[0]
        x2 = tree.children[1].children[0]

        assert x == x2

        y = tree.children[0].children[1]
        z = tree.children[1].children[1]

        op1 = Tree("mul", [])
        op2 = Tree("add", [])

        op2.children = [y, z]
        op1.children = [x, op2]

        tree = op1

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0].children[0]
        x2 = entire.children[1].children[0]

        assert x == x2

        y = entire.children[0].children[1]
        z = entire.children[1].children[1]

        op1 = Tree("mul", [])
        op2 = Tree("add", [])

        op2.children = [y, z]
        op1.children = [x, op2]

        entire = op1

        return tree, entire


def times_identity_l(tree, entire):

    """"Performs x * 1 = x
    Give the * as cursor
    """
    # one = Tree("suc", [Token("NUMBER", "0")])

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        tree = tree.children[0]

        grandparent.children[treeindex] = tree
        return grandparent, entire

    else:

        entire = entire.children[0]

        return tree, entire


def times_identity_r(tree, entire):
    """"Performs  x = x * 1
    Give the x as cursor
    """
    mul = Tree("mul", [])
    one = Tree("suc", [Token("NUMBER", "0")])

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        # Check this
        mul.children = [tree, one]
        tree = mul

        grandparent.children[treeindex] = tree
        return grandparent, entire

    else:

        mul.children = [entire, one]
        entire = mul

        return tree, entire


def power_of_one_l(tree, entire):

    """"Performs 1 ^ x = 1
    Give the * as cursor
    """
    one = Tree("suc", [Token("NUMBER", "0")])

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        tree = one

        grandparent.children[treeindex] = tree
        return grandparent, entire

    else:

        entire = one

        return tree, entire


def first_power_of_x_l(tree, entire):

    """"Performs x ^ 1 = x
    Give the ^ as cursor
    """
    # one = Tree("suc", [Token("NUMBER", "0")])

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        tree = tree.children[0]

        grandparent.children[treeindex] = tree
        return grandparent, entire

    else:

        entire = entire.children[0]

        return tree, entire


def first_power_of_x_r(tree, entire):

    """"Performs   x = x ^ 1
    Give the x as cursor
    """
    one = Tree("suc", [Token("NUMBER", "0")])
    power = Tree("pow", [])
    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        power.children = [tree, one]
        tree = power

        grandparent.children[treeindex] = tree
        return grandparent, entire

    else:

        power.children = [tree, one]
        entire = power

        return tree, entire


def distributivity_power_plus_l(tree, entire):

    """"
    Performs x ^ (y + z) = x ^ y * x ^ z
    """
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0]
        y = tree.children[1].children[0]
        z = tree.children[1].children[1]
        x2 = copy.deepcopy(x)

        op1 = Tree("pow", [])
        op2 = Tree("pow", [])
        op3 = Tree("mul", [])

        op1.children = [x, y]
        op2.children = [x2, z]
        op3.children = [op1, op2]

        tree = op3

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0]
        y = entire.children[1].children[0]
        z = entire.children[1].children[1]
        x2 = copy.deepcopy(x)

        op1 = Tree("pow", [])
        op2 = Tree("pow", [])
        op3 = Tree("mul", [])

        op1.children = [x, y]
        op2.children = [x2, z]
        op3.children = [op1, op2]

        entire = op3

        return tree, entire


def distributivity_power_plus_r(tree, entire):
    """"
    Performs   x ^ y * x ^ z = x ^ (y + z)
    """
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0].children[0]
        x2 = tree.children[1].children[0]

        assert x == x2

        y = tree.children[0].children[1]
        z = tree.children[1].children[1]

        op1 = Tree("pow", [])
        op2 = Tree("add", [])

        op2.children = [y, z]
        op1.children = [x, op2]

        tree = op1

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0].children[0]
        x2 = entire.children[1].children[0]

        assert x == x2

        y = entire.children[0].children[1]
        z = entire.children[1].children[1]

        op1 = Tree("pow", [])
        op2 = Tree("add", [])

        op2.children = [y, z]
        op1.children = [x, op2]

        entire = op1

        return tree, entire


def distributivity_power_times_l(tree, entire):

    """"
    Performs (x * y) ^ z = x ^ z * x ^ z
    """
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        z = tree.children[1]
        x = tree.children[0].children[0]
        y = tree.children[0].children[1]
        z2 = copy.deepcopy(z)

        op1 = Tree("pow", [])
        op2 = Tree("pow", [])
        op3 = Tree("mul", [])

        op1.children = [x, z]
        op2.children = [y, z2]
        op3.children = [op1, op2]

        tree = op3

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        z = entire.children[1]
        x = entire.children[0].children[0]
        y = entire.children[0].children[1]
        z2 = copy.deepcopy(z)

        op1 = Tree("pow", [])
        op2 = Tree("pow", [])
        op3 = Tree("mul", [])

        op1.children = [x, z]
        op2.children = [y, z2]
        op3.children = [op1, op2]

        entire = op3

        return tree, entire


def distributivity_power_times_r(tree, entire):
    """"
    Performs  x ^ z * x ^ z = (x * y) ^ z
    """
    # TODO Check this with the axioms

    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0].children[0]
        z = tree.children[0].children[1]

        y = tree.children[1].children[0]
        z2 = tree.children[1].children[1]
        assert z == z2

        op1 = Tree("pow", [])
        op2 = Tree("mul", [])

        op2.children = [x, y]
        op1.children = [op2, z]

        tree = op1

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0].children[0]
        z = entire.children[0].children[1]

        y = entire.children[1].children[0]
        z2 = entire.children[1].children[1]
        assert z == z2

        op1 = Tree("pow", [])
        op2 = Tree("mul", [])

        op2.children = [x, y]
        op1.children = [op2, z]

        entire = op1

        return tree, entire


def distributivity_power_power_l(tree, entire):
    """"
    Performs (x ^ y) ^ z= x ^ (y * z)
    """
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0].children[0]
        y = tree.children[0].children[1]
        z = tree.children[1]

        op1 = Tree("mul", [])
        op2 = Tree("pow", [])

        op1.children = [y, z]
        op2.children = [x, op1]

        tree = op2

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0].children[0]
        y = entire.children[0].children[1]
        z = entire.children[1]

        op1 = Tree("mul", [])
        op2 = Tree("pow", [])

        op1.children = [y, z]
        op2.children = [x, op1]

        entire = op2

        return tree, entire


def distributivity_power_power_r(tree, entire):
    """"
    Performs  x ^ (y * z)  = (x ^ y) ^ z
    """
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent

        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        x = tree.children[0]
        y = tree.children[1].children[0]
        z = tree.children[1].children[1]

        op1 = Tree("pow", [])
        op2 = Tree("pow", [])

        op1.children = [x, y]
        op2.children = [op1, z]

        tree = op2

        grandparent.children[treeindex] = tree

        return tree, entire

    else:

        x = entire.children[0]
        y = entire.children[1].children[0]
        z = entire.children[1].children[1]

        op1 = Tree("pow", [])
        op2 = Tree("pow", [])

        op1.children = [x, y]
        op2.children = [op1, z]

        entire = op2

        return tree, entire
