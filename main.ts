// Sokoban solver: push-optimal, room by room.
//
// One program answers a whole set of levels (5 instances under /input/<id>/)
// by writing /output/<id>/solution.txt for each: the player's moves in LURD
// (lowercase; the checker replays them and counts the steps that push a box).
//
// Every generated level is a 1-wide hallway with 1-4 rooms off it, one door
// each; a box pushed through a door into the hallway can never come back, so
// each room is an independent problem and the level's push-optimal count is
// the sum of the rooms' optima. The solver therefore:
//
//   1. cuts the hallway's dead squares (floor of the player's row from which
//      no push can ever reach a goal) out of the floor and flood-fills the
//      rest: each part holding boxes is one room;
//   2. solves each room push-optimally with A* over pushes (a state is the
//      boxes plus the player's reachable region; the heuristic is the
//      minimum-cost matching of boxes to goals by shortest-path distance),
//      pruning dead squares and frozen 2x2 blocks, the other rooms' boxes
//      standing as walls; the player walks on from where the last room left
//      it, so the walks between rooms come out of the same search;
//   3. falls back to solving the whole level with the same search when the
//      level does not decompose, and writes an empty (valid, 0-point) answer
//      for an instance no search solves within budget.
//
// Deterministic: no clock, no randomness; every budget is a count of
// expanded states, every tie is broken by insertion order, so two runs write
// the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** First A* budget per room (expanded states); a room that fails is retried at RETRY_BUDGET. */
const BUDGET = 150_000;
const RETRY_BUDGET = 1_000_000;
/** Budget for the whole-level fallback search. */
const WHOLE_BUDGET = 1_000_000;

const DIRS: readonly (readonly [number, number, string])[] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box alone can be pushed onto some goal (pulled back from the goals over open floor). */
function liveSquares(w: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Shortest walking distance from every square to each goal, over floor alone (walls block, boxes ignored). */
function goalDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array[] {
  return goals.map((g) => {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && dist[q] < 0) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    return dist;
  });
}

/**
 * Minimum-cost matching of boxes to goals by the precomputed distances
 * (subset DP; n <= 4 here). Infinity when some box reaches no goal.
 * Admissible: every box must be pushed at least its matched distance.
 */
function matching(boxes: number[], dists: Int32Array[], goals: number[]): number {
  const n = boxes.length;
  const full = (1 << n) - 1;
  const dp = new Float64Array(1 << n).fill(Infinity);
  dp[0] = 0;
  for (let mask = 0; mask <= full; mask++) {
    if (dp[mask] === Infinity) continue;
    const i = popcount(mask); // next box to place
    if (i >= n) continue;
    for (let j = 0; j < n; j++) {
      if (mask & (1 << j)) continue;
      const d = dists[j]![boxes[i]!];
      if (d < 0) continue;
      const nm = mask | (1 << j);
      const v = dp[mask]! + d;
      if (v < dp[nm]!) dp[nm] = v;
    }
  }
  return dp[full]!;
}

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

/** A binary heap ordered by (f, -g, insertion order): lowest first, deterministic. */
class Heap {
  private f: number[] = [];
  private g: number[] = [];
  private n: number[] = [];
  private c: number[] = [];
  get size(): number {
    return this.f.length;
  }
  push(f: number, g: number, node: number, counter: number): void {
    this.f.push(f);
    this.g.push(g);
    this.n.push(node);
    this.c.push(counter);
    let i = this.f.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      this.swap(i, p);
      i = p;
    }
  }
  pop(): number {
    const top = this.n[0]!;
    const last = this.f.length - 1;
    this.f[0] = this.f[last]!;
    this.g[0] = this.g[last]!;
    this.n[0] = this.n[last]!;
    this.c[0] = this.c[last]!;
    this.f.pop();
    this.g.pop();
    this.n.pop();
    this.c.pop();
    let i = 0;
    for (;;) {
      let child = 2 * i + 1;
      if (child >= this.f.length) break;
      if (child + 1 < this.f.length && this.less(child + 1, child)) child++;
      if (!this.less(child, i)) break;
      this.swap(i, child);
      i = child;
    }
    return top;
  }
  private less(a: number, b: number): boolean {
    return (
      this.f[a]! < this.f[b]! ||
      (this.f[a] === this.f[b] &&
        (this.g[a]! > this.g[b]! || (this.g[a] === this.g[b] && this.c[a]! < this.c[b]!)))
    );
  }
  private swap(a: number, b: number): void {
    [this.f[a], this.f[b]] = [this.f[b]!, this.f[a]!];
    [this.g[a], this.g[b]] = [this.g[b]!, this.g[a]!];
    [this.n[a], this.n[b]] = [this.n[b]!, this.n[a]!];
    [this.c[a], this.c[b]] = [this.c[b]!, this.c[a]!];
  }
}

interface SearchResult {
  pushes: number;
  moves: string;
  expanded: number;
}

/**
 * Push-optimal A* for one (sub-)level: the player starts at `player`, the
 * boxes at `boxes`, the goals at `goals`, on board `wall`. Returns the
 * push-optimal {pushes, moves} or null when the expansion budget runs out
 * (or the level is unsolvable). `expandedOut` reports work done.
 */
function solvePushAStar(
  w: number,
  wall: Uint8Array,
  goals: number[],
  boxes0: number[],
  player0: number,
  maxExpansions: number,
  expandedOut: { n: number },
): SearchResult | null {
  const size = wall.length;
  const n = boxes0.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(w, wall, goals);
  const dists = goalDistances(w, wall, goals);
  const solved = (bs: number[]): boolean => bs.every((b) => goalSet[b]);
  if (boxes0.some((b) => !live[b])) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size); // region flood-fill stamps
  const mark = new Int32Array(size); // region-low flood-fill stamps
  const stack = new Int32Array(size);
  let stamp = 0;

  /** Lowest square of the region reachable from `from` around walls and boxes; -1 when `from` is blocked. */
  const regionLow = (from: number, stampArr: Int32Array, st: number): number => {
    if (wall[from] || occupied[from]) return -1;
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    stampArr[from] = st;
    while (sp > 0) {
      const p = stack[--sp]!;
      if (p < low) low = p;
      const up = p - w;
      const dn = p + w;
      const lf = p - 1;
      const rt = p + 1;
      if (!wall[up] && !occupied[up] && stampArr[up] !== st) {
        stampArr[up] = st;
        stack[sp++] = up;
      }
      if (!wall[dn] && !occupied[dn] && stampArr[dn] !== st) {
        stampArr[dn] = st;
        stack[sp++] = dn;
      }
      if (!wall[lf] && !occupied[lf] && stampArr[lf] !== st) {
        stampArr[lf] = st;
        stack[sp++] = lf;
      }
      if (!wall[rt] && !occupied[rt] && stampArr[rt] !== st) {
        stampArr[rt] = st;
        stack[sp++] = rt;
      }
    }
    return low;
  };

  /** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
  const frozenBlock = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b]) || (occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
    }
    return false;
  };

  // State key: region low, then the sorted boxes. A string key stays exact
  // for any box count (a numeric base-`size` key would overflow 2**53 on a
  // whole level with many boxes).
  const keyOf = (bs: number[], low: number): string => low + ':' + bs.join(',');

  const boxesList: number[][] = [boxes0.slice()];
  const playerList: number[] = [player0];
  const gScore: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const pushDir: number[] = [-1];

  const h0 = matching(boxes0, dists, goals);
  if (h0 === Infinity) return null;
  const seenKey = new Map<string, number>();
  for (const b of boxes0) occupied[b] = 1;
  const low0 = regionLow(player0, seen, ++stamp);
  for (const b of boxes0) occupied[b] = 0;
  if (low0 < 0) return null;
  seenKey.set(keyOf(boxes0, low0), 0);

  const heap = new Heap();
  let counter = 0;
  heap.push(h0, 0, 0, counter++);
  let expanded = 0;
  let found = solved(boxes0) ? 0 : -1;

  while (heap.size > 0 && found < 0) {
    const s = heap.pop();
    // A stale heap entry: a better g for this key was recorded since.
    const bs = boxesList[s]!;
    const pl = playerList[s]!;
    for (const b of bs) occupied[b] = 1;
    const low = regionLow(pl, seen, ++stamp);
    if (low < 0 || seenKey.get(keyOf(bs, low)) !== s) {
      for (const b of bs) occupied[b] = 0;
      continue;
    }
    if (expanded >= maxExpansions) {
      for (const b of bs) occupied[b] = 0;
      break;
    }
    expanded++;
    const g = gScore[s]!;
    const here = stamp; // this state's region stamp in `seen`
    for (let bi = 0; bi < n; bi++) {
      const b = bs[bi]!;
      for (let dir = 0; dir < 4; dir++) {
        const [dx, dy] = DIRS[dir]!;
        const d = dy * w + dx;
        const t = b + d;
        // The player must stand on b - d to push, and t must be free floor.
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(t);
        let key = '';
        let next: number[] | null = null;
        let h = Infinity;
        if (!frozen) {
          next = bs.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          const nlow = regionLow(b, mark, ++stamp);
          if (nlow >= 0) {
            key = keyOf(next, nlow);
            h = matching(next, dists, goals);
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null || key === '' || h === Infinity) continue;
        const prev = seenKey.get(key);
        if (prev !== undefined && gScore[prev]! <= g + 1) continue;
        const ns = boxesList.length;
        seenKey.set(key, ns);
        boxesList.push(next);
        playerList.push(b); // after the push the player stands on the box's old square
        gScore.push(g + 1);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        pushDir.push(dir);
        if (solved(next)) {
          found = ns;
          break;
        }
        heap.push(g + 1 + h, g + 1, ns, counter++);
      }
      if (found >= 0) break;
    }
    for (const b of bs) occupied[b] = 0;
  }
  expandedOut.n = expanded;
  if (found < 0) return null;

  // Reconstruct: the pushes in order, then the walks between them.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = player0;
  let boxes = boxes0.slice();
  for (const s of path) {
    const nextBoxes = boxesList[s]!;
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const dir = pushDir[s]!;
    const d = to - from;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += DIRS[dir]![2];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves, expanded };
}

/** Shortest walk from `from` to `to` around walls and boxes (u, d, l, r order on ties). */
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room. Null when a box starts in the hallway, a goal lies
 * outside every part, or some part holds more or fewer boxes than goals.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve one level: room by room when it decomposes, whole-level otherwise. */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  const stat = { n: 0 };
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    for (const room of rooms) {
      // The other rooms' boxes stand as walls (on their goals once solved).
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      let solved = solvePushAStar(level.w, wall, room.goals, room.boxes, player, BUDGET, stat);
      if (solved === null) solved = solvePushAStar(level.w, wall, room.goals, room.boxes, player, RETRY_BUDGET, stat);
      if (solved === null) return '';
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      room.done = true;
    }
    return moves;
  }
  const solved = solvePushAStar(level.w, level.wall, level.goals, level.boxes, level.player, WHOLE_BUDGET, stat);
  return solved === null ? '' : solved.moves;
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
