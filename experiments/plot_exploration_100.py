import re
import cv2
import glob
import argparse
import numpy as np
from colour import Color


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Plot exploration_100 experiment result as heatmap')
    arg_parser.add_argument('--maze-size', type=str, default='10x10',
                            help='Size of maze, default 10x10')
    arg_parser.add_argument('--steps', type=int, default=5,
                            help='Number of steps at the beginning to record')
    arg_parser.add_argument('--fig', type=str, default='vis_exploration.png',
                            help='Filename for saved plot figure')
    arg_parser.add_argument(
        '--episode-folder-pattern', '-p', type=str, default='',
        help='Filename pattern for episode log folders')
    args = arg_parser.parse_args()
    return args


def collect_agent_pos(log_file_pattern, steps=None):
    log_files = glob.glob(log_file_pattern)
    pos_dict = {}
    total_pos = 0
    for log in log_files:
        collect_steps = 0
        with open(log, 'r') as f:
            for l in f.readlines():
                if l.startswith('agent:'):
                    if steps is not None and collect_steps >= steps:
                        break
                    pos = re.search('\[.*\]', l)
                    pos_str = pos.group(0).replace(', ', '_')[1:-1]
                    if pos_str in pos_dict.keys():
                        pos_dict[pos_str] += 1
                    else:
                        pos_dict[pos_str] = 1

                    collect_steps += 1

                    total_pos += 1

    return pos_dict, total_pos


def plot_exploration_heatmap(pos_dict, total_pos, log_file_pattern,
                             maze_size, fig):
    maze_size = [float(i) for i in maze_size.split('x')]
    example_log = glob.glob(log_file_pattern)[0]
    init_ob = cv2.imread(example_log.replace('log.txt', 'init_ob.png'))
    init_ob = cv2.resize(init_ob, (80, 80))  # for better offset
    dark_red = Color("#550000")
    colors = list(dark_red.range_to(Color("#ffaaaa"), int(total_pos)))
    colors.reverse()
    h, w, _ = init_ob.shape
    for pos_str, count in pos_dict.items():
        pos = [int(i) for i in pos_str.split('_')]
        y = int(pos[0] * h / maze_size[0])
        x = int(pos[1] * w / maze_size[1])
        y_ = int((pos[0] + 1) * h / maze_size[0])
        x_ = int((pos[1] + 1) * w / maze_size[1])
        color = np.array(colors[count].rgb[::-1]) * 255
        init_ob[y:y_, x:x_] = np.asarray(color, dtype=np.uint8)

    cv2.imwrite(fig, init_ob)
    print('Saved {}'.format(fig))


def main():
    args = parse_args()
    agent_pos_statistic = collect_agent_pos(args.episode_folder_pattern,
                                            steps=args.steps)
    print(agent_pos_statistic)
    plot_exploration_heatmap(*agent_pos_statistic, args.episode_folder_pattern,
                             args.maze_size, args.fig)


if __name__ == '__main__':
    main()
