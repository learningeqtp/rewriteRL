import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lark import Tree, Token
from rewriterl.env import Robinson
from rewriterl.load_parse import define_levels, load_dataset
from rewriterl.tree_ops import insert_cursor_node, winning_state
from torch.distributions import Categorical

class TreeNNCursor2(nn.Module):
    
    def __init__(self, embedding_dim=5, internal_dim=2, bilinear_mul=False):
        """"
        Initialize, we need a value for the zero token and for each of the possible transformations
        """
        super(TreeNNCursor2, self).__init__()
        self.token = nn.Parameter(torch.randn(embedding_dim).to(device), requires_grad=False)
        # self.add = nn.Linear(2*embedding_dim, embedding_dim)
        self.add = nn.Sequential(
            nn.Linear(2 * embedding_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, embedding_dim),
        )
        self.bimul = bilinear_mul
        if bilinear_mul:
            self.mul = nn.Bilinear(embedding_dim, embedding_dim, embedding_dim)
        else:
            # self.mul = nn.Linear(2 * embedding_dim, embedding_dim)
            self.mul = nn.Sequential(
                nn.Linear(2 * embedding_dim, internal_dim),
                nn.ReLU(),
                nn.Linear(internal_dim, embedding_dim),
            )
        # Bilinear layer
        # self.suc = nn.Linear(embedding_dim, embedding_dim)
        self.suc = nn.Sequential(
            nn.Linear(embedding_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, embedding_dim),
        )
        # self.cursor = nn.Linear(embedding_dim, embedding_dim)
        self.cursor = nn.Sequential(
            nn.Linear(embedding_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, embedding_dim),
        )

    def forward(self, inputs):

        """Input will be a Tree object

        """

        if type(inputs) == Tree:
            if inputs.data == "suc":
                # print("Suc")
                return self.suc(self.forward(inputs.children[0]))

            elif inputs.data == "cursor":
                # print("cursor")
                return self.cursor(self.forward(inputs.children[0]))

            elif inputs.data == "mul":
                # print("Mul")
                if self.bimul:
                    #
                    return self.mul(
                        self.forward(inputs.children[0]),
                        self.forward(inputs.children[1]),
                    )

                else:
                    return self.mul(
                        torch.cat(
                            (
                                self.forward(inputs.children[0]),
                                self.forward(inputs.children[1]),
                            )
                        )
                    )

            elif inputs.data == "add":
                # print("Add")
                return self.add(
                    torch.cat(
                        (
                            self.forward(inputs.children[0]),
                            self.forward(inputs.children[1]),
                        )
                    )
                )

        elif type(inputs) == Token:
            # print("Token")
            return self.token


class Policy(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Policy, self).__init__()

        self.actions = np.arange(action_space)
        self.action_space = action_space
        self.state_processor = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)
        # self.sigmoid = nn.Sigmoid()
        self.embedding = TreeNNCursor2(
            embedding_dim=state_space, internal_dim=state_space
        )

    def forward(self, state):

        embedded_state = self.embedding(state)
        # print(embedded_state)
        out = torch.relu(self.state_processor(embedded_state))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)

        return out

    def action(self, state):
        """
        Samples the action based on their probability
        """

        action_prob = self.forward(state)
        # print(action_prob)
        probs = torch.softmax(action_prob, dim=-1)
        probs = probs.clone()
        moves = env.legal_indices()

        if 9 in moves:
            moves.remove(9)
        # print(moves)
        # Epsilon-greedy policy
        epsilon = np.random.uniform(0, 1, 1)
        if epsilon > 0.05:
            for i in range(9):
                if not i in moves:
                    probs[i] *= 0

            probs /= sum(probs)
            # print(probs)
            m = Categorical(probs)
            # print(m)
            action = m.sample()
            # print(action)
        else:
            action = torch.tensor(random.choice(moves), dtype=torch.long).reshape(1)

        if action == 9:
            raise ValueError
        return action

    def greedy_action(self, state):
        """
        Returns the greedy action
        """

        action_prob = self.forward(state)
        probs = torch.softmax(action_prob, dim=-1)
        probs = probs.clone()
        moves = env.legal_indices()
        if 9 in moves:
            moves.remove(9)

        for i in range(9):
            if not i in moves:
                probs[i] *= 0
        probs /= sum(probs)

        action = torch.argmax(probs)
        return action

class ReplayBuffer:
    def __init__(self, mem=1):
       

        self.problem_dict = {k[3]: [] for k in total_problems}
        self.buffer = []
        self.memsize = mem

    def add_sample(self, states, actions, rewards, problem):
        episode = {
            "states": states,
            "actions": actions,
            "problem": problem,
            "rewards": rewards,
            "summed_rewards": sum(rewards),
        }

        self.problem_dict[episode["problem"]].append(episode)


    def sort(self):

        for problem in total_problems:
            problem_number = problem[3]
            self.problem_dict[problem_number] = [
                k for k in self.problem_dict[problem_number] if k["summed_rewards"] > 0
            ]
            # Only take shortest proof
            self.problem_dict[problem_number] = sorted(
                self.problem_dict[problem_number],
                key=lambda x: len(x["states"]),
                reverse=False,
            )

            self.problem_dict[problem_number] = self.problem_dict[problem_number][:self.memsize]

        new_buffer = []
        for problem in total_problems:
            problem_number = problem[3]
            new_buffer = new_buffer + self.problem_dict[problem_number]

        self.buffer = new_buffer

    def get_random_samples(self, batch_size):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        return batch

    def __len__(self):
        return len(self.buffer)

def draw_transition(saved_episode):

    T = len(saved_episode["states"])
    t1 = np.random.randint(0, T - 1)

    return t1


def create_training_input(episode, t1):

    state = episode["states"][t1]
    action = episode["actions"][t1]
    return state, action


def create_training_examples(batch_size):

    input_list = []
    output_array = []

    episodes = buffer.get_random_samples(batch_size)
    for ep in episodes:

        t1 = draw_transition(ep)

        state, action = create_training_input(ep, t1)
        input_list.append([state])
        output_array.append(action)
    return input_list, output_array


def train_policy(batch_size):

    X, Y = create_training_examples(batch_size)

    total_batch_loss = 0

    optimizer.zero_grad()
    for s in range(batch_size):
        state = X[s][0]

        y = Y[s].reshape(1)
        y_ = model(state).reshape(1, 9)


        pred_loss = F.cross_entropy(y_, y)
        total_batch_loss += pred_loss

    total_batch_loss.backward()

    optimizer.step()

    return total_batch_loss.item()


def evaluate(problems, episode_length=100, greedy_action=False):

    sample, value, difficulty, index = random.choice(problems)
    env.load_problem_tree(sample, val=value)

    rewards = 0
    for j in range(episode_length):
        nn_cursor, nn_state = env.get_deepcopy_state_variables()
        nn_state = insert_cursor_node(nn_cursor, nn_state)
        if greedy_action:
            action = model.greedy_action(nn_state)
        else:
            action = model.action(nn_state)
        cursor, state, done = env.step(action)
        reward = int(done)
        rewards += reward
        if done:
            break
    return rewards

def generate_episode(problem, episode_length=100):

    sample, value, difficulty, index = problem

    env.load_problem_tree(sample, val=value)

    states = []
    actions = []
    rewards = []
    for j in range(episode_length):

        nn_cursor, nn_state = env.get_deepcopy_state_variables()

        nn_state = insert_cursor_node(nn_cursor, nn_state)
        states.append(nn_state)
        action = model.action(nn_state)
        cursor, state, done = env.step(action)

        reward = int(done)

        actions.append(action)
        rewards.append(reward)

        if done:
            nn_cursor, nn_state = env.get_deepcopy_state_variables()
            nn_state = insert_cursor_node(nn_cursor, nn_state)
            states.append(nn_state)
            break

    return [states, actions, rewards, index]



def run_training(config, epochs=100, expname='test', server=0, episode_length=100):
    """
    """

    
    import os
    if not os.path.exists('/outputfolder/results'):
        os.makedirs('/outputfolder/results')

    if not os.path.exists('/outputfolder/results/model_parameters'):
        os.makedirs('/outputfolder/results/model_parameters')

    with open(f'/outputfolder/results/{expname}', 'a') as f:
        for line in config:
            f.write(line + "\n")

    name = f"/outputfolder/results/model_parameters/model_{expname}.pth"

    torch.save(model.state_dict(), name)
    level = 0
    problems = total_problems[: (400 * (level + 1))]
    all_rewards = []
    losses = []
    average_100_reward = []

    # Reporting:

    result_list = []
    for ep in range(1, epochs + 1):

        print(ep, len(problems))

        loss_buffer = []




        for eval in range(n_episodes_per_iter):
            print("Generating new experience {}/{}".format(eval + 1, n_episodes_per_iter))
            problem = random.choice(problems)

            generated_episode = generate_episode(problem, episode_length=episode_length)
            buffer.add_sample(
                generated_episode[0],
                generated_episode[1],
                generated_episode[2],
                generated_episode[3],

            )

        buffer.sort()

        print("Current amount of problems in buffer: {}".format(len(buffer.buffer)))
        for eval in range(n_updates_per_iter):
            print("Updating behavior function {}/{}".format(eval + 1, n_updates_per_iter))
            ac_loss = train_policy(batch_size)
            loss_buffer.append(ac_loss)
        ac_loss = np.mean(loss_buffer)
        losses.append(ac_loss)

        reward_list = []

        for eval in range(n_evals):
            print("Evaluating on {}/{}".format(eval + 1, n_evals))
            ep_rewards = evaluate(problems, episode_length=episode_length, greedy_action=True)
            reward_list.append(ep_rewards)

        all_rewards.append(np.mean(reward_list))
        average_100_reward.append(np.mean(all_rewards[-100:]))
        if all_rewards[-1] > 0.95:
            level += 1
            problems = total_problems[: (400 * (level + 1))]


        print(
            "\rEpisode: {} | Rewards: {:.2f} | Mean_100_Rewards: {:.2f} | Loss: {:.2f} | No. Problems {}".format(
                ep,
                np.mean(reward_list),
                np.mean(all_rewards[-100:]),
                ac_loss,
                len(problems),
            ),
            end="",
            flush=True,
        )
        reporting = [str(ep), str(np.mean(reward_list)),str(level), str(len(problems)), str(ac_loss)]
        result_list.append(reporting)
        name = f"model_parameters/model_{expname}.pth"






        import os
        if not os.path.exists('/outputfolder/results'):
            os.makedirs('/outputfolder/results')

        if not os.path.exists('/outputfolder/results/model_parameters'):
            os.makedirs('/outputfolder/results/model_parameters')

        with open(f'/outputfolder/results/{expname}', 'a') as f:
            f.write(",".join(reporting) + "\n")

        name = f"/outputfolder/results/model_parameters/model_{expname}.pth"

        torch.save(model.state_dict(), name)

    return (
        all_rewards,
        average_100_reward,
        losses,
    )

if __name__ == "__main__":


    import sys

    import argparse

    dataset = load_dataset("train")
    levels = define_levels(dataset, no_lopl=False)
    env = Robinson()

    action_space = 9

    total_problems = []

    for i in range(len(levels)):
        total_problems += levels[i]

    # Filter problems

    total_problems = [k for k in total_problems if not winning_state(k[0])]
    total_problems = [k for k in total_problems if k[2] <= 89]
    print(len(total_problems))

    startup_problems = total_problems[: (400 * (1))]

    device = "cpu"

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-w', '--warmup', help='Warmup eps', required=True, type=int)
    parser.add_argument('-b', '--batches', help='Batches per iter', required=True, type=int)
    parser.add_argument('-e', '--episodes', help='Episodes per iter', required=True, type=int)
    parser.add_argument('-v', '--evals', help='Evals per iter', required=True, type=int)
    parser.add_argument('-s', '--batchsize', help='Batch size', required=True, type=int)
    parser.add_argument('-o', '--hidden', help='Number of hidden units per iter', required=True, type=int)
    parser.add_argument('-l', '--learningrate', help='Learning rate', required=True, type=float)
    parser.add_argument('-d', '--duration', help='Episode length/duration', required=True, type=int)
    parser.add_argument('-p', '--epochs', help='Number of epochs', required=True, type=int)
    parser.add_argument('-q', '--statespace', help='Size of embedding space', required=True, type=int)
    parser.add_argument('-n', '--expname', help='Batches per iter', required=True)
    parser.add_argument('-z', '--server', help='On server?', required=True, type=int)
    parser.add_argument('-m', '--memory', help='How many memories to keep of each problem', required=True, type=int)

    args = parser.parse_args()
    n_warm_up_episodes = args.warmup
    n_updates_per_iter = args.batches
    n_episodes_per_iter = args.episodes
    n_evals = args.evals
    batch_size = args.batchsize
    hidden_units = args.hidden
    learning_rate = args.learningrate
    episode_length = args.duration
    epochs = args.epochs
    state_space = args.statespace
    expname = args.expname
    server = args.server
    memory = args.memory

    configs = [device, n_warm_up_episodes, n_updates_per_iter,
               n_episodes_per_iter, n_evals, batch_size, hidden_units,
               learning_rate, episode_length, epochs, expname, memory, state_space]
    config_keys = ["Device", "Warmup Size", "Batches per epoch", "New episodes per epoch",
                   "Number of evals", "Batch size", "Hidden Units", "learning rate", "episode length",
                   "Number of epochs", "Experiment name", "No buffer items per problem", "State space"]

    configs = [str(k) for k in configs]
    zipped_config = [key + " " + value for key, value in zip(config_keys, configs)]

    buffer = ReplayBuffer(mem=memory)
    model = Policy(state_space, action_space, hidden_units).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    samples = []

    buffer.sort()
    print("Current amount of problems in buffer: {}".format(len(buffer.buffer)))


    # Call this file using something like:
    # "nohup python 3sil_example.py --warmup 1000 --batches 250 --episodes 1000 --evals 400 --batchsize 32 --hidden 64 --learningrate 0.001 --duration 100 --epochs 250 --statespace 16 --expname S1P1R_stored_1  --server 1 --memory 1 > log.txt &"
