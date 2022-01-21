from time import sleep

from tqdm import trange


def wait(s):
    for _ in trange(s - 1, bar_format='{n_fmt}/{total_fmt} seconds', initial=1, total=s):
        sleep(1)


wait(5)
