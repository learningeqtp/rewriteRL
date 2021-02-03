from rewriterl.load_parse import get_dataset
import numpy as np
from rewriterl.tree_ops import (
    pydot__tree_to_png,
    check_tree,
    label_parents,
    winning_state,
    legal_actions,
    legal_actions_indices,
    plus_zero_l,
    plus_zero_r,
    recursive_plus_l,
    recursive_plus_r,
    times_zero_l,
    recursive_times_l,
    recursive_times_r,
    count_cursors,
    calculate_value,
    calculate_value_full,
)
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
    poly_legal_actions,
    poly_legal_actions_indices,
    tree_poly_png,
)
from lark import tree, Token, Tree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import imageio
import glob
import os
import copy
from pathlib import Path

here = Path(__file__).parent


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


class RobinsonPoly:
    def __init__(self, generation=False):

        if not os.path.exists("images"):
            os.makedirs("images")
        files = glob.glob("images/*")
        for f in files:
            os.remove(f)

        self.generation = generation

    def load_problem_tree(self, tree, goal):

        """"
        Load problem by state
        """
        self.stepcounter = 0

        self.state = copy.deepcopy(tree)
        self.goal = goal
        # TODO Write a new check_tree method
        # assert check_tree(self.state)
        if self.state == self.goal:
            raise ValueError("Starting state of problem is already winning state")

        self.state = label_parents(self.state)
        self.cursor = self.state

    def get_simulator(self):
        """"
        Returns a (deep) copy of the environment with the cursor at the same place

        What you do in the simulator should not affect the 'real' environment
        """

        path = self.cursor_location_from_root()
        sim_env = RobinsonPoly()

        sim_env.load_problem_tree(self.state, self.goal)
        for step in path:
            sim_env.cursor = sim_env.cursor.children[step]

        return sim_env

    def get_deepcopy_state_variables(self):
        """"
        Return a real copy of the state and the cursor (with interconnected reference)
        """
        path = self.cursor_location_from_root()
        new_state = copy.deepcopy(self.state)
        new_state = label_parents(new_state)
        new_cursor = new_state
        # print(path)
        for step in path:
            new_cursor = new_cursor.children[step]

        return new_cursor, new_state

    def cursor_location_from_root(self):

        """"
        Go upwards from cursor and save the path to get back there
        Note: might not be super efficient
        """

        path = []
        traveler = self.cursor
        while traveler is not self.state:
            travelercopy = traveler
            traveler = traveler.parent
            for e, child in enumerate(traveler.children):
                if child is travelercopy:
                    path.append(e)

        return list(reversed(path))

    def legal_indices(self):

        indices = poly_legal_actions_indices(self.cursor)

        return indices

    def step(self, action):

        current_legal = self.legal_indices()

        if not action in current_legal:

            raise AssertionError("Action is not legal")

        if action == 0:
            _, self.state = plus_zero_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 1:
            _, self.state = plus_zero_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 2:
            _, self.state = recursive_plus_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 3:
            _, self.state = recursive_plus_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 4:
            _, self.state = times_zero_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 5:
            _, self.state = recursive_times_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 6:
            _, self.state = recursive_times_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 7:

            self.cursor = self.cursor.children[0]

        elif action == 8:

            self.cursor = self.cursor.children[1]

        elif action == 9:
            self.cursor.children[1], self.cursor.children[0] = (
                self.cursor.children[0],
                self.cursor.children[1],
            )
            # TODO
            # Think about if the cursor needs to go to root after switching the subtrees
            self.cursor = self.state

        elif action == 10:
            _, self.state = power_zero_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 11:
            _, self.state = recursive_power_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 12:
            _, self.state = recursive_power_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 13:
            _, self.state = associativity_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 14:

            _, self.state = associativity_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 15:
            _, self.state = distributivity__times_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 16:
            _, self.state = distributivity__times_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 17:
            _, self.state = times_identity_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 18:
            _, self.state = times_identity_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 19:
            _, self.state = power_of_one_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 20:
            _, self.state = first_power_of_x_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 21:
            _, self.state = first_power_of_x_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 22:
            _, self.state = distributivity_power_plus_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 23:
            _, self.state = distributivity_power_plus_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 24:
            _, self.state = distributivity_power_times_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 25:
            _, self.state = distributivity_power_times_r(self.cursor, self.state)

            self.cursor = self.state

        elif action == 26:
            _, self.state = distributivity_power_power_l(self.cursor, self.state)

            self.cursor = self.state

        elif action == 27:
            _, self.state = distributivity_power_power_r(self.cursor, self.state)

            self.cursor = self.state

        self.state = label_parents(self.state)
        self.state.parent = None

        self.stepcounter += 1
        if self.state == self.goal:
            winning = True
        else:
            winning = False

        return self.cursor, self.state, winning

    def visualize(self):

        plt.axis("off")
        tree_poly_png(
            self.cursor,
            self.state,
            "images/{}.png".format(self.stepcounter),
            rankdir="TB",
        )

        img = mpimg.imread("images/{}.png".format(self.stepcounter))
        imgplot = plt.imshow(img)
        plt.show()

    def visualize_loc(self, loc):
        plt.axis("off")
        tree_poly_png(
            self.cursor, self.state, f"images/{loc}.png", rankdir="TB",
        )

        img = mpimg.imread(f"images/{loc}.png")
        imgplot = plt.imshow(img)
        plt.show()

    def visualize_goal(self):

        plt.axis("off")
        tree_poly_png(
            self.cursor, self.goal, "images/goal.png", rankdir="TB",
        )

        img = mpimg.imread("images/{}.png".format(self.stepcounter))
        imgplot = plt.imshow(img)
        plt.show()

    def make_gif(self):

        filenames = glob.glob("images/*")

        images = []

        filenames = sorted(filenames, key=lambda x: int(x[7:-4]))
        print(filenames)
        for filename in filenames:
            images.append(imageio.imread(filename))

        print(len(images))

        imageio.mimsave("gifs/f.gif", images, format="GIF", duration=0.8)
        im4 = imageio.get_reader("gifs/f.gif")
        writer = imageio.get_writer("movie.gif", duration=0.8)
        for im in im4:
            writer.append_data(im[:, :, :])
        writer.close()
