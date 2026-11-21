# Sokoban Solver — push-optimal, room by room

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![make.fun · rank](https://make.fun/badge/sokoban-solver-01-run-20261121/arifsaputra03/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![make.fun · reward](https://make.fun/badge/sokoban-solver-01-run-20261121/arifsaputra03/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A Sokoban solver for the make.fun competition
`sokoban-solver-01-run-20261121`. `main.ts` (Node 22, standard library
only) answers a whole set of 5 instances in one run, writing
`/output/<id>/solution.txt` for each: the player's moves in LURD
(lowercase `u/d/l/r`).

Every generated level is a 1-wide hallway with 1–4 rooms off it, and a box
pushed through a door into the hallway can never come back, so each room is
an independent problem and the level's push-optimal count is the sum of its
rooms' optima. The solver therefore decomposes each level into rooms and
solves every room push-optimally with A* over pushes (state = boxes plus
the player's reachable region; heuristic = minimum-cost matching of boxes
to goals by shortest-path distance), pruning dead squares and frozen 2×2
blocks, and falls back to solving the whole level when it does not
decompose.

On the kit's practice set it scores **1,000,000** — every instance solved
in exactly its push-optimal count (13, 16, 29, 47, 60 pushes) — against
**200,000** for the house baseline plain-astar 1.0.0. The whole set runs in
about 0.4 s, far inside the 60 s wall budget, and two runs write
byte-identical output (no clock, no randomness; every budget is a count of
expanded states).

Run it locally:

```sh
MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<output> node main.ts
```

## License

MIT — see `LICENSE`.
