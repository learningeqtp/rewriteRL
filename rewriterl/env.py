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


class Robinson:
    def __init__(self, generation=False):

        if not os.path.exists("images"):
            os.makedirs("images")
        files = glob.glob("images/*")
        for f in files:
            os.remove(f)

        self.generation = generation

    def load_problem_tree(self, tree, val=-1):

        """"
        Load problem by state
        """
        self.stepcounter = 0
        self.value = val
        self.state = copy.deepcopy(tree)
        if not check_tree(self.state):
            raise AssertionError("Tree is not correct (check_tree method fails)")

        # assert check_tree(self.state)
        if not self.generation:
            if winning_state(self.state):
                raise ValueError("Starting state of problem is already winning state")

        self.state = label_parents(self.state)
        self.cursor = self.state

    def get_simulator(self):
        """"
        Returns a (deep) copy of the environment with the cursor at the same place

        What you do in the simulator should not affect the 'real' environment
        """

        path = self.cursor_location_from_root()
        sim_env = Robinson()

        sim_env.load_problem_tree(self.state, val=self.value)
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

    def legal(self):

        return legal_actions(self.cursor)

    def legal_indices(self, associativity=False):
        # TODO: check what should be the default
        indices = legal_actions_indices(self.cursor)

        # The agent is not able to make use of the fact that
        # * and +  are associative by switching the subtrees
        if not associativity:
            if 9 in indices:
                indices.remove(9)
            assert not (9 in indices)

        return indices

    def step(self, action, return_env=False):

        current_legal = legal_actions(self.cursor)

        if not current_legal[action] == 1:
            raise AssertionError("Action is not legal")

        if action == 0:

            _, self.state = plus_zero_l(self.cursor, self.state)
            # reset cursor to root

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

        if not check_tree(self.state):
            raise AssertionError("Check_tree failed")

        # Check if the tree still has the same value!
        # If not -- something major is wrong!

        assert calculate_value_full(self.state) == self.value
        # print(self.state)
        # print(calculate_value_full(self.state))
        # print(self.value)

        self.stepcounter += 1
        winning = winning_state(self.state)

        self.state = label_parents(self.state)
        self.state.parent = None

        if winning:
            if not self.value == -1:
                assert calculate_value(self.state) == self.value

        return self.cursor, self.state, winning

    def visualize(self):

        plt.axis("off")
        pydot__tree_to_png(
            self.cursor,
            self.state,
            "images/{}.png".format(self.stepcounter),
            rankdir="TB",
        )

        img = mpimg.imread("images/{}.png".format(self.stepcounter))
        imgplot = plt.imshow(img)
        plt.show()

    def make_gif(self):

        filenames = glob.glob("images/*")
        # print(filenames)
        images = []

        filenames = sorted(filenames, key=lambda x: int(x[7:-4]))
        print(filenames)
        for filename in filenames:
            images.append(imageio.imread(filename))

        print(len(images))

        imageio.mimsave("gifs/f.gif", images, format="GIF", duration=0.2)
        im4 = imageio.get_reader("gifs/f.gif")
        writer = imageio.get_writer("movie.gif", duration=0.2)
        for im in im4:
            writer.append_data(im[:, :, :])
        writer.close()

    def random_walk(self, steps):

        self.visualize()

        for i in range(steps):
            oldstate = repr(self.state)
            legal = legal_actions(self.cursor)

            indices = [k for k in range(10) if legal[k] == 1]
            rand_action = random.choice(indices)
            self.cursor, self.state, winning = self.step(rand_action)
            if hasattr(self.state, "data"):
                self.visualize()
            if winning:

                return winning

            if rand_action is not 7 and rand_action is not 8 and rand_action is not 9:
                if oldstate == repr(self.state):
                    break

    def random_walk_generator(self, steps):

        # self.visualize()

        generated_goals = []
        generated_goals_dup = []
        for i in range(steps):
            legal = legal_actions_indices(self.cursor)
            # print(legal)
            if 9 in legal:
                legal.remove(9)
            if 1 in legal and len(legal) > 1:
                if np.random.uniform(0, 1, 1) < 0.7:
                    legal.remove(1)

            rand_action = random.choice(legal)
            self.cursor, self.state, winning = self.step(rand_action)
            # if i == 50:
            #     self.visualize()
            # Last move was not just a cursor movement means the tree changed
            already_seen = any([k == self.state for k in generated_goals_dup])
            if not already_seen:
                generated_goals.append([copy.deepcopy(self.state), self.value, i + 1])
                generated_goals_dup.append(copy.deepcopy(self.state))
        return generated_goals
