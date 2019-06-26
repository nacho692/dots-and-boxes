import sys
from dots_boxes import DotsAndBoxes

env = DotsAndBoxes(2)
env.render()

inp = ""
while inp != "quit":
    inp = sys.stdin.readline()


    def splitter(n):
        node = list(map(int, n.split(",")))
        return node[0], node[1]


    action = list(map(splitter, inp.split(" ")))
    observation, reward, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()