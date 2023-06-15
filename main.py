import argparse

from game import Runner

TRAPS = [(2, 2), (3, 3), (4, 4), (5, 5)]
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
START = (3, 0)
GOAL = (3, 7)

parser = argparse.ArgumentParser(description='My script')
parser.add_argument('-a', '--actions', help='Number of actions that agent can take (4, 8, 9')
parser.add_argument('-e', '--episodes', help='Number of training episodes')
parser.add_argument('-v', '--verbose', help='verbose')
parser.add_argument('-t', '--traps', help='Number of traps 0,1,2,3,4')

args = parser.parse_args()
actions_list = [4, 8, 9]
if args.actions:
    actions_list = [int(args.actions)]
episodes = int(args.episodes or 200)
verbose = int(args.verbose or 10)
traps = min(4, int(args.traps or 0))
for actions in actions_list:
    if actions not in [4, 8, 9]:
        raise Exception('Invalid "-a/--actions"')

for actions in actions_list:
    game = Runner(num_actions=actions,
                  start=START,
                  goal=GOAL,
                  verbose=verbose,
                  episodes=episodes,
                  rows=7,
                  cols=10,
                  gamma=1,
                  wind=WIND,
                  traps=TRAPS[:traps],
                  render_interval=[51, 50])
    game.train()
    game.test(pause=3)
