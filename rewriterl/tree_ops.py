from lark import Visitor, Tree, Transformer, v_args, tree
from lark import Token
import copy


def is_token(term):

    from lark.lexer import Token

    if type(term) == Token:
        return True
    else:
        return False


def pre_order_traversal(tree):

    res = []
    if type(tree) == Tree:
        res.append(tree.data)
    else:
        res.append(tree.value)

    if type(tree) == Tree:

        res += pre_order_traversal(tree.children[0])

        if len(tree.children) > 1:

            res += pre_order_traversal(tree.children[1])

    return res


# From lark package
def pydot__tree_to_png(cursor, tree, filename, rankdir="LR", **kwargs):
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


# From lark package
def pydot__tree_to_png_cursornode(tree, filename, rankdir="TB", **kwargs):
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

    def _to_pydot(subtree):
        color = hash(subtree.data) & 0xFFFFFF
        color |= 0x808080

        subnodes = [
            _to_pydot(child) if isinstance(child, Tree) else new_leaf(child)
            for child in subtree.children
        ]

        if subtree.data == "cursor":
            node = pydot.Node(i[0], style="filled", fillcolor="red", label=subtree.data)
        elif subtree.data == "add":
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
        else:
            node = pydot.Node(
                i[0], style="filled", fillcolor="#%x" % color, label=subtree.data
            )
        i[0] += 1
        graph.add_node(node)

        for subnode in subnodes:
            graph.add_edge(pydot.Edge(node, subnode))

        return node

    _to_pydot(tree)
    graph.write_png(filename)


def count_cursors(node):

    cursors = 0

    if hasattr(node, "cursor"):
        if node.cursor == True:
            print("YES")
            cursors = cursors + 1
        else:
            if type(tree) == Tree:
                if len(tree.children) > 0:
                    for e, child in enumerate(tree.children):
                        cursors = cursors + count_cursors(child)

    return cursors


def label_parents(tree):

    if hasattr(tree, "children"):
        for child in tree.children:
            child.parent = tree

            if type(child) == Tree:
                if len(child.children) > 0:
                    label_parents(child)

    return tree


def get_size(tree):

    size = 0

    if type(tree) == Tree or type(tree) == Token:
        size += 1

    if type(tree) == Tree:

        if len(tree.children) == 1:
            size += get_size(tree.children[0])

        elif len(tree.children) == 2:
            size += get_size(tree.children[0])
            size += get_size(tree.children[1])

    return size


def construct_goal_state(value):

    if value == 0:
        root = Tree("cursor", [Token("NUMBER", "0")])
        return root
    elif value > 0:
        root = Tree("cursor", [Tree("suc", [])])
        tree = root
        for i in range(value):
            tree.children = [Tree("suc", [])]
            tree = tree.children[0]

        tree.children = [Token("NUMBER", "0")]
        return root


def construct_integer(value):

    if value == 0:
        root = Token("NUMBER", "0")
        return root
    elif value > 0:
        root = Tree("suc", [])
        tree = root
        for i in range(1, value):
            tree.children = [Tree("suc", [])]
            tree = tree.children[0]

        tree.children = [Token("NUMBER", "0")]
        return root


def calculate_value_full(tree):

    val = 0

    if type(tree) == Token:
        return 0
    elif type(tree) == Tree:

        if tree.data == "suc":

            val = 1 + calculate_value_full(tree.children[0])

        elif tree.data == "add":
            val = calculate_value_full(tree.children[0]) + calculate_value_full(
                tree.children[1]
            )
        elif tree.data == "mul":
            val = calculate_value_full(tree.children[0]) * calculate_value_full(
                tree.children[1]
            )

    return val


def calculate_value(tree):

    """"Calculate the value of the tree at the end to make sure the algo doesn't get
    to cheat
    # TODO replace this function with the full version above
    """
    val = 0
    # print("_----------")
    if type(tree) == Token:
        return 0
    if type(tree) == Tree:
        if tree.data == "suc":
            val += 1
        # print(val)

        for e, child in enumerate(tree.children):
            val += calculate_value(child)

        return val


def insert_cursor_node(tree, entire):

    if hasattr(tree, "parent") and tree.parent is not None:

        gp = tree.parent
        # copy = tree
        cursor_tag = Tree("cursor", [])
        cursor_tag.children = [tree]

        for e, child in enumerate(gp.children):
            # print(child, tree, e)
            if child is tree:
                gp.children[e] = cursor_tag

        return entire
    else:
        cursor_tag = Tree("cursor", [])
        cursor_tag.children = [tree]

        return cursor_tag


def winning_state(tree):
    """"
    This functions travels through the tree and if there are only succesor ops, the tree is in winning state
    """

    if not (type(tree) == Tree or type(tree) == Token):
        raise ValueError(
            "This is not a tree, cannot calculate whether tree is in winning state"
        )
    winning = True

    if hasattr(tree, "data"):
        if tree.data != "suc":
            winning = False

            if not winning:
                return winning

        for child in tree.children:

            winning = winning_state(child)
            if not winning:
                return winning

    return winning


def check_tree(tree):
    """"
    Checks the tree for correctness: number of children, operations, etc. This should always return True, otherwise the
    tree is malformed.
    """
    wellformed = True

    if hasattr(tree, "data"):
        if tree.data == "suc":
            wellformed = len(tree.children) < 2

        elif tree.data == "add" or tree.data == "mul":
            wellformed = len(tree.children) == 2
    else:
        return True

    for child in tree.children:
        # Only have to look deeper if we haven't found a fault yet
        if wellformed == True:
            if type(child) == Tree:
                if len(child.children) > 0:
                    wellformed = check_tree(child)
                    if not wellformed:
                        break
    return wellformed


def plus_zero_l(tree, entire):

    """"Performs x + 0 = x
    Give the "+" as root
    """
    nonzero = tree.children[0]
    # print(nonzero)
    if hasattr(tree, "parent") and tree.parent is not None:
        grandparent = tree.parent
        for e, child in enumerate(grandparent.children):

            if child is tree:
                grandparent.children[e] = nonzero

        return grandparent, entire

    else:

        entire = nonzero

        return tree, entire


def plus_zero_r(tree, entire):
    """"
    Performs x = x + 0
    """

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent
        for e, child in enumerate(grandparent.children):
            if child is tree:
                treeindex = e

        add = Tree("add", [])
        zerotoken = Token("NUMBER", "0")
        add.children = [tree, zerotoken]

        grandparent.children[treeindex] = add

        return tree, entire
    else:

        add = Tree("add", [])
        zerotoken = Token("NUMBER", "0")
        add.children = [tree, zerotoken]
        entire = add
        return tree, entire


def recursive_plus_l(tree, entire):
    """"Performs the recursively defined addition
    x + s(y) = s(x+y)

    First argument should be the + tree, second the entire tree (same if these coincide)
    """

    #
    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent
        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        suc_op = tree.children[1]

        # This reference [cop] was unnecessary and confusing
        # cop = tree

        rc = suc_op.children[0]
        lc = tree.children[0]
        tree.children = [lc, rc]

        # This reference [cop] was unnecessary and confusing
        # Keeping it in comments for reference
        # suc_op.children = [cop]

        suc_op.children = [tree]
        grandparent.children[treeindex] = suc_op

        return grandparent, entire

    else:

        cop = entire
        suc = cop.children[1]

        cop.children = [cop.children[0], suc.children[0]]
        suc.children = [cop]
        entire = suc

        return tree, entire


def recursive_plus_r(tree, entire):
    """"
    Implements s(x + y) -> x + s(y)
    """

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent
        for e, child in enumerate(grandparent.children):

            if child is tree:
                treeindex = e

        succop = tree

        new_root = tree.children[0]

        succop.children[0] = tree.children[0].children[1]
        new_root.children[1] = succop

        grandparent.children[treeindex] = new_root

        return tree, entire
    else:

        succop = entire
        newroot = entire.children[0]
        succop.children[0] = entire.children[0].children[1]
        newroot.children[1] = succop
        entire = newroot

        return tree, entire


def times_zero_l(tree, entire):

    """"Performs x * 0 = 0
    Give the * as root
    """
    zero = tree.children[1]

    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent
        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                grandparent.children[e] = zero

        return grandparent, entire

    else:
        # print("Here")
        entire = zero
        # print(tree)
        return tree, entire


def recursive_times_l(tree, entire):
    """" Performs x * s(y) = x * y + x

    Supply the cursor at *

    """
    if hasattr(tree, "parent") and tree.parent is not None:

        grandparent = tree.parent
        for e, child in enumerate(grandparent.children):
            # print(child, tree, e)
            if child is tree:
                treeindex = e

        # Make the new add node
        add = Tree("add", [])

        # This cop seems unnecessary too
        # Leaving it for reference
        # cop = tree

        grandparent.children[treeindex] = add
        # Copy the x term that will be in both multiply and add op
        x = copy.deepcopy(tree.children[0])

        # Move the child over the S() to the y
        tree.children[1] = tree.children[1].children[0]

        # Give the add node its children
        add.children = [tree, x]

        return grandparent, entire

    else:
        # Need a new op, the add
        add = Tree("add", [])
        # Cop is unnecessary
        # cop = entire

        x = copy.deepcopy(entire.children[0])
        entire.children[1] = entire.children[1].children[0]
        add.children = [entire, x]
        entire = add

        return tree, entire


def recursive_times_r(tree, entire):
    """"
    Performs
    x * y + x -> x * s(y)
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


def legal_actions(cursor):
    """"
    This function returns a list of which actions are legal
    """

    legal = [0] * 10  # 4 actions + 2 cursor movements

    if hasattr(cursor, "data"):
        if cursor.data == "add" and repr(cursor.children[1]) == "Token(NUMBER, '0')":
            legal[0] = 1

        legal[1] = 1

        if cursor.data == "add":
            if hasattr(cursor.children[1], "data"):

                if cursor.children[1].data == "suc":
                    legal[2] = 1

        if cursor.data == "suc":
            if hasattr(cursor.children[0], "data"):
                if cursor.children[0].data == "add":
                    legal[3] = 1

        if cursor.data == "mul" and repr(cursor.children[1]) == "Token(NUMBER, '0')":
            legal[4] = 1

        if cursor.data == "mul":

            if hasattr(cursor.children[1], "data"):

                if cursor.children[1].data == "suc":
                    legal[5] = 1

        if cursor.data == "add":
            if type(cursor.children[0]) == Tree:
                if cursor.children[0].data == "mul":
                    if cursor.children[0].children[0] == cursor.children[1]:
                        legal[6] = 1

        if type(cursor.children[0]) == Tree:
            legal[7] = 1

        if len(cursor.children) > 1:
            if type(cursor.children[1]) == Tree:
                legal[8] = 1

        if len(cursor.children) > 1:
            legal[9] = 1

    # TODO: Perhaps remove this, as it was for goal-oriented rewriting mainly
    elif hasattr(cursor, "value"):
        legal[1] = 1
    return legal


def legal_actions_goal_rewriting(cursor):
    """"
    This function returns a list of which actions are legal
    """

    legal = [0] * 10  # 4 actions + 2 cursor movements

    if hasattr(cursor, "data"):
        if cursor.data == "add" and repr(cursor.children[1]) == "Token(NUMBER, '0')":
            legal[0] = 1

        legal[1] = 1

        if cursor.data == "add":
            if hasattr(cursor.children[1], "data"):

                if cursor.children[1].data == "suc":
                    legal[2] = 1

        if cursor.data == "suc":
            if hasattr(cursor.children[0], "data"):
                if cursor.children[0].data == "add":
                    legal[3] = 1

        if cursor.data == "mul" and repr(cursor.children[1]) == "Token(NUMBER, '0')":
            legal[4] = 1

        if cursor.data == "mul":

            if hasattr(cursor.children[1], "data"):

                if cursor.children[1].data == "suc":
                    legal[5] = 1

        if cursor.data == "add":
            if type(cursor.children[0]) == Tree:
                if cursor.children[0].data == "mul":
                    if cursor.children[0].children[0] == cursor.children[1]:
                        legal[6] = 1

        if type(cursor.children[0]) == Tree or type(cursor.children[0]) == Token:
            legal[7] = 1

        if len(cursor.children) > 1:
            if type(cursor.children[0]) == Tree or type(cursor.children[0]) == Token:
                legal[8] = 1

        if len(cursor.children) > 1:
            legal[9] = 1

    elif hasattr(cursor, "value"):
        legal[1] = 1

    return legal


def legal_actions_indices(cursor):

    legal = legal_actions(cursor)
    indices = [k for k in range(10) if legal[k] == 1]
    return indices


def legal_actions_indices_goal_rewriting(cursor):

    legal = legal_actions_goal_rewriting(cursor)
    indices = [k for k in range(10) if legal[k] == 1]
    return indices
