"""
csmvt_model.py — Model-only extraction from Cost_Sensitive_MVT.ipynb
Auto-generated: keeps full definitions (functions/classes/dataclasses) and safe imports/constants.
Top-level demo/plotting/CLI code is omitted.
"""
# foraging_sim.py
# Explorer → Forager on a shared DataFrame; no 'unknown' anywhere.
# Explorer reveals mines via per-mine 'revealed' flags; auto-labels when fully revealed / empty.
# Forager softmax decisions + animations that reflect live, shared env.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import colors as mcolors
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import matplotlib.patches as mpatches

# ------------------- Color map for environment richness -------------------
RICHNESS_COLORS = {
    "poor": "sandybrown",
    "neutral": "lightgreen",
    "rich": "gold",
}

# ------------------- Environment initializer (TRUE has no 'unknown') -------------------
def init_gridworld(
    size: int = 3,
    seed: Optional[int] = None,
    # TRUE overall richness distribution
    p_overall: List[float] = (0.10, 0.35, 0.55),  # poor, neutral, rich
    # P(#mines = 0..3)
    p_mines: List[float] = (0.25, 0.40, 0.25, 0.10),
    # Optional preset visible map: dict[(row,col)->label] to override labels (e.g., a known map)
    preset_visible_overall: Optional[Dict[Tuple[int,int], str]] = None,
) -> pd.DataFrame:
    """
    Builds an N×N grid with:
      - TRUE hidden fields: overall∈{poor,neutral,rich}, mines∈{poor,neutral,rich} or None
      - VISIBLE fields start labeled (from preset if provided, else from TRUE).
      - Mines start HIDDEN using explicit 'Mine i revealed' flags (False where a TRUE mine exists).
        Visible mine fields are None until revealed by the explorer.
    """
    rng = np.random.default_rng(seed if seed is not None else None)

    true_overall_labels = np.array(["poor", "neutral", "rich"])
    p_overall = np.asarray(p_overall, dtype=float)
    p_overall = p_overall / p_overall.sum()

    p_mines = np.asarray(p_mines, dtype=float); p_mines /= p_mines.sum()

    mine_cat_given_overall = {
        "poor":    {"poor": 0.70, "neutral": 0.25, "rich": 0.05},
        "neutral": {"poor": 0.25, "neutral": 0.50, "rich": 0.25},
        "rich":    {"poor": 0.10, "neutral": 0.30, "rich": 0.60},
    }
    reward_prob_map = {"poor": 0.20, "neutral": 0.50, "rich": 0.80}

    def sample_k(ov: str) -> int:
        base = p_mines.copy()
        if ov == "rich":
            base[:2] = 0.0   # at least 2 mines
        elif ov == "poor":
            base[3] = 0.0   # cannot have 3 mines
        s = base.sum()
        if s == 0:
            return 2 if ov == "rich" else (1 if ov == "poor" else 0)
        base /= s
        return int(rng.choice([0, 1, 2, 3], p=base))

    def sample_mine_cat(ov: str, k: int) -> List[Optional[str]]:
        if k == 0:
            return [None, None, None]
        probs_map = mine_cat_given_overall[ov]
        cats = np.array(["poor", "neutral", "rich"])
        probs = np.array([probs_map["poor"], probs_map["neutral"], probs_map["rich"]], dtype=float)
        probs = probs / probs.sum()
        sampled = rng.choice(cats, size=k, p=probs).tolist()
        return (sampled + [None] * (3 - k))[:3]

    digs_range = {"poor": (1, 2), "neutral": (2, 4), "rich": (3, 6)}
    def digs_allowed_for(cat: Optional[str]) -> Optional[int]:
        if cat is None: return None
        low, high = digs_range[cat]
        return int(rng.integers(low, high + 1))

    N = size
    center = (N // 2, N // 2)
    rows = []

    for r in range(N):
        for c in range(N):
            is_center = (r, c) == center
            if is_center:
                overall_true = "poor"   # base is empty poor
                t1 = t2 = t3 = None
                rp1 = rp2 = rp3 = None
                d1 = d2 = d3 = None
                env_empty_true = True
            else:
                overall_true = rng.choice(true_overall_labels, p=p_overall).item()
                k = sample_k(overall_true)
                t1, t2, t3 = sample_mine_cat(overall_true, k)
                rp1 = reward_prob_map[t1] if t1 else None
                rp2 = reward_prob_map[t2] if t2 else None
                rp3 = reward_prob_map[t3] if t3 else None
                d1 = digs_allowed_for(t1)
                d2 = digs_allowed_for(t2)
                d3 = digs_allowed_for(t3)
                env_empty_true = (k == 0)

            rows.append({
                "Location": f"{r}:{c}", "Row": r, "Col": c, "is_center": is_center,

                # TRUE (no 'unknown' anywhere)
                "TRUE Number 1 mines": t1,
                "TRUE Number 2 mines": t2,
                "TRUE Number 3 mines": t3,
                "TRUE Mine 1 reward_prob": rp1,
                "TRUE Mine 2 reward_prob": rp2,
                "TRUE Mine 3 reward_prob": rp3,
                "TRUE Mine 1 digs_allowed": d1,
                "TRUE Mine 2 digs_allowed": d2,
                "TRUE Mine 3 digs_allowed": d3,
                "TRUE env_empty": env_empty_true,
                "TRUE Overall Richness": overall_true,

                # VISIBLE initial mines are hidden (revealed flags False where a TRUE mine exists)
                "Number 1 mines": None,
                "Number 2 mines": None,
                "Number 3 mines": None,
                "Mine 1 reward_prob": None,
                "Mine 2 reward_prob": None,
                "Mine 3 reward_prob": None,
                "Mine 1 digs_allowed": None,
                "Mine 2 digs_allowed": None,
                "Mine 3 digs_allowed": None,

                "Mine 1 revealed": (t1 is None),
                "Mine 2 revealed": (t2 is None),
                "Mine 3 revealed": (t3 is None),

                # Runtime
                "Overall Richness of this environment": None,  # set below
                "Explorer Label": None, 
                "env_empty": env_empty_true,
                "visited_by_explorer": False,
            })

    env = pd.DataFrame(rows)

    # --- Visible overall richness: use PRESET if provided; otherwise TRUE ---
    if preset_visible_overall:
        env["Overall Richness of this environment"] = env["TRUE Overall Richness"]
        for (rr, cc), lab in preset_visible_overall.items():
            env.loc[(env["Row"] == rr) & (env["Col"] == cc),
                    "Overall Richness of this environment"] = lab
    else:
        env["Overall Richness of this environment"] = env["TRUE Overall Richness"]

    return env

# ------------------- Runtime helpers (operate IN-PLACE on the shared env) -------------------
def ensure_runtime_columns(env: pd.DataFrame) -> pd.DataFrame:
    # Keep digs_remaining columns
    for i in (1, 2, 3):
        digs_col = f"Mine {i} digs_allowed"
        rem_col  = f"Mine {i} digs_remaining"
        if rem_col not in env.columns:
            env[rem_col] = env[digs_col]
        else:
            mask = env[rem_col].isna() & env[digs_col].notna()
            env.loc[mask, rem_col] = env.loc[mask, digs_col]

    # Ensure revealed flags exist (if missing, infer from visible mine fields)
    for i in (1, 2, 3):
        rev_col = f"Mine {i} revealed"
        if rev_col not in env.columns:
            # revealed if visible category is not None OR TRUE mine absent
            #env[rev_col] = env[f"Number {i} mines"].notna() | env[f"TRUE Number {i} mines"].isna()
            env[rev_col] = False

    if "visited_by_explorer" not in env.columns:
        env["visited_by_explorer"] = False
    if "env_empty" not in env.columns:
        env["env_empty"] = False
    return env  # same object

def _indexify(env: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have a MultiIndex on (Row, Col) while KEEPING columns."""
    if list(env.index.names) != ["Row", "Col"]:
        env.set_index(["Row", "Col"], inplace=True, drop=False)
    return env

def _cell_total_remaining_digs_df(env_idx: pd.DataFrame, pos: Tuple[int, int]) -> int:
    s = 0
    for i in (1, 2, 3):
        rem = env_idx.loc[pos, f"Mine {i} digs_remaining"]
        if pd.notna(rem): s += int(float(rem))
    return s

# ------------------- Leader -------------------
@dataclass
class LeaderConfig:
    total_food: int = 200 #leader_set_explorer
    #if past one round - team_reward = ExplorerAgent.total_food_left() + MVTAgent.total_food_left() + MVTAgent.total_reward
    alpha: float = 0.3

class LeaderAgent:
    """
    - Random tradeoff of the total_food to explorer and forager
    - Based on coins collected after the round * learning_rate, allocated differently proportionally
    """
    def __init__(self, cfg: LeaderConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed if seed is not None else None)

        self.threshold = True 
        self.resource_tradeoff = [0.5 * self.cfg.total_food, 0.5 * self.cfg.total_food]

     #def tradeoff(self):
        #randomly selected
        #proportions = np.random.rand(1,2)

         #start with half-half...
    
    def set_forager(self, forager: "MVTAgent"):
        self.forager = forager
        
    def update_allocation(self): #after each game
        #the leader should increase forage resouces because explorer is based on luck 
        #perceived_reward = self.forager.total_reward (if round 1) or self.team_reward (if more than one round) * self.cfg.alpha 
        explorer_resources, forager_resources = self.resource_tradeoff
        perceived_reward = self.forager.total_food_left()
        diff_in_rewards = perceived_reward - self.forager.total_reward

        change = (self.cfg.alpha * diff_in_rewards) / self.forager.total_reward

        explorer_resources *= (1 - change)
        forager_resources *= (1 + change)

        total = explorer_resources + forager_resources
        scale = self.cfg.total_food / total
        explorer_resources *= scale
        forager_resources *= scale
     
        self.resource_tradeoff = [explorer_resources, forager_resources]
        return self.resource_tradeoff

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Assumed available from your codebase:
# - ensure_runtime_columns(env_df): adds runtime cols like visited_by_explorer, env_empty, etc.
# - _indexify(env_df): ensures MultiIndex by (row, col)

# ------------------- Explorer (count-only: #mines & revealed flags) + CSV logging -------------------
@dataclass
class ExplorerConfig:
    init_resource: int = 100
    move_cost: int = 4
    scan_cost: int = 2
    gamma: float = 0.1
    beta_local: float = 0.6     # weight on local hidden-mine pressure
    beta_global: float = 0.4    # weight on global hidden-mine pressure
    avoid_base: bool = True
    no_backtrack: bool = True   # never revisit an already visited block
    cost_sensitive_index: int = 0.1

class ExplorerAgent:
    """
    COUNT-ONLY EXPLORER (CSV-logging version)
    - Decisions depend ONLY on:
        (1) how many TRUE mine slots exist on tiles, and
        (2) whether each slot is revealed (Mine i revealed == True/False).
    - Single scan per cell (reveals ONE hidden mine slot).
    - No backtracking (cannot enter a previously visited cell).
    - Opens tiles for the Forager by setting env['visited_by_explorer'] = True.
    - Auto-label rules:
        * After first visit: if tile truly has NO mines -> label 'poor' + mark all three as revealed.
        * When ALL TRUE mines at the tile are revealed -> set label to TRUE overall and mark all revealed.
    - Logging:
        * Every step logs action, decision values, feasibility flags, resources, position, counts, and config.
        * Call run(..., csv_path="explorer_round1.csv") to persist the log as CSV.
    - Transfer map to Forager: export_env_for_forager() returns the SAME env reference.
    """
    def __init__(self, env_df: pd.DataFrame, cfg: ExplorerConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed if seed is not None else None)

        self.env = ensure_runtime_columns(env_df)
        _indexify(self.env)

        self.resource = cfg.init_resource
        size = int(self.env.index.get_level_values(0).max()) + 1
        self.base = (size // 2, size // 2)
        self.pos = self.base
        self.left_base_once = False
        self.t = 0
        self.total_moves = 0
        self.log: List[Dict] = []

        # Open base for the forager and possibly autolabel
        self.env.loc[self.pos, 'visited_by_explorer'] = True
        self._autolabel_if_ready(self.pos)

        # Enforce single scan per cell
        self.scanned_once: set[Tuple[int, int]] = set()

        # Animation buffers (unchanged)
        self.frames_pos: List[Tuple[int, int]] = []
        self.frames_resource: List[float] = []
        self.frames_action: List[str] = []
        self.frames_unrevealed_mask: List[np.ndarray] = []
        self.frames_reward_dummy: List[float] = []
        self.frames_decision: List[str] = []
        self._snapshot("start", "start")

    # ---- utilities ----
    def _neighbors(self):
        r, c = self.pos
        neigh = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        valid = [(rr, cc) for rr, cc in neigh if (rr, cc) in self.env.index]
        if self.cfg.avoid_base and self.left_base_once:
            valid = [p for p in valid if p != self.base]
        if self.cfg.no_backtrack:
            valid = [p for p in valid if not bool(self.env.loc[p, 'visited_by_explorer'])]
        return valid

    def _hidden_mines_here(self) -> List[int]:
        """Return mine IDs among {1,2,3} that truly exist and are not yet revealed here."""
        mines = []
        for i in (1, 2, 3):
            has_true = self.env.loc[self.pos, f"TRUE Number {i} mines"] is not None
            revealed = bool(self.env.loc[self.pos, f"Mine {i} revealed"])
            if has_true and (not revealed):
                mines.append(i)
        return mines

    def _global_hidden_count(self) -> int:
        """Total number of hidden TRUE mine slots across the whole grid."""
        cnt = 0
        for pos in self.env.index:
            for i in (1, 2, 3):
                has_true = self.env.loc[pos, f"TRUE Number {i} mines"] is not None
                revealed = bool(self.env.loc[pos, f"Mine {i} revealed"])
                if has_true and (not revealed):
                    cnt += 1
        return cnt

    def _fully_revealed(self, pos: Tuple[int, int]) -> bool:
        """Fully revealed if every TRUE mine slot at pos has revealed=True."""
        for i in (1, 2, 3):
            has_true = self.env.loc[pos, f"TRUE Number {i} mines"] is not None
            if has_true and (not bool(self.env.loc[pos, f"Mine {i} revealed"])):
                return False
        return True

    def _autolabel_if_ready(self, pos: Tuple[int, int]) -> None:
        """
        Refresh 'Overall Richness of this environment' and 'Explorer Label' if:
          - explorer has visited this pos, AND
          - (a) tile has no TRUE mines -> label 'poor' and mark all revealed
          - (b) OR all TRUE mines at the tile are revealed -> set to TRUE overall and mark all revealed.
        """
        if not bool(self.env.loc[pos, "visited_by_explorer"]):
            return

        has_true_mine = any(self.env.loc[pos, f"TRUE Number {i} mines"] is not None for i in (1, 2, 3))
        if not has_true_mine:
            self.env.loc[pos, "Overall Richness of this environment"] = "poor"
            self.env.loc[pos, "Explorer Label"] = "poor"
            for i in (1, 2, 3):
                self.env.loc[pos, f"Mine {i} revealed"] = True
            return

        if self._fully_revealed(pos):
            self.env.loc[pos, "Overall Richness of this environment"] = self.env.loc[pos, "TRUE Overall Richness"]
            self.env.loc[pos, "Explorer Label"] = self.env.loc[pos, "TRUE Overall Richness"]
            for i in (1, 2, 3):
                self.env.loc[pos, f"Mine {i} revealed"] = True

    def _snapshot(self, action: str, decision: str):
        idx = self.env
        nrows = int(idx.index.get_level_values(0).max()) + 1
        ncols = int(idx.index.get_level_values(1).max()) + 1

        mask = np.zeros((nrows, ncols), dtype=bool)
        for pos in idx.index:
            r, c = pos
            unrevealed = (not bool(idx.loc[pos, "visited_by_explorer"])) or any(
                (idx.loc[pos, f"TRUE Number {i} mines"] is not None) and (not bool(idx.loc[pos, f"Mine {i} revealed"]))
                for i in (1, 2, 3)
            )
            mask[r, c] = unrevealed

        self.frames_unrevealed_mask.append(mask)
        self.frames_pos.append(tuple(self.pos))
        self.frames_resource.append(float(self.resource))
        self.frames_action.append(action)
        self.frames_decision.append(decision)
        self.frames_reward_dummy.append(0.0)

    # ---------- rich CSV logger ----------
    def _log_step(
        self,
        *,
        action: str,
        decision: str,
        resource_before: float,
        resource_after: float,
        v_stay: Optional[float],
        v_leave: Optional[float],
        can_scan_here: Optional[bool],
        can_move: Optional[bool],
        mine_id: Optional[int] = None,
        revealed_cat: Optional[str] = None,
        move_probs: Optional[List[float]] = None
    ):
        row = {
            # time & pos
            "step": self.t,
            "row": self.pos[0],
            "col": self.pos[1],
            "action": action,
            "decision": decision,

            # resources
            "resource_before": float(resource_before),
            "resource_after": float(resource_after),

            # decision values & feasibility
            "v_stay": (None if v_stay is None else float(v_stay)),
            "v_leave": (None if v_leave is None else float(v_leave)),
            "can_scan_here": (None if can_scan_here is None else bool(can_scan_here)),
            "can_move_neighbors": (None if can_move is None else bool(can_move)),

            # counts
            "hidden_local": int(len(self._hidden_mines_here())),
            "hidden_global": int(self._global_hidden_count()),
            "total_moves": int(self.total_moves),

            # environment labels at current cell
            "visited_here": bool(self.env.loc[self.pos, "visited_by_explorer"]),
            "overall_label": str(self.env.loc[self.pos, "Overall Richness of this environment"]),
            "explorer_label": str(self.env.loc[self.pos, "Explorer Label"]),

            # action-specific
            "mine_id": (None if mine_id is None else int(mine_id)),
            "revealed_cat": (None if revealed_cat is None else str(revealed_cat)),
            "move_probs": (None if move_probs is None else list(move_probs)),

            # config snapshot (so the CSV is self-contained)
            "cfg_init_resource": int(self.cfg.init_resource),
            "cfg_move_cost": int(self.cfg.move_cost),
            "cfg_scan_cost": int(self.cfg.scan_cost),
            "cfg_gamma": float(self.cfg.gamma),
            "cfg_beta_local": float(self.cfg.beta_local),
            "cfg_beta_global": float(self.cfg.beta_global),
            "cfg_avoid_base": bool(self.cfg.avoid_base),
            "cfg_no_backtrack": bool(self.cfg.no_backtrack),
        }
        self.log.append(row)

    # ---- core actions (count-only) ----
    def _reveal_one_mine_here(self, v_stay: Optional[float], v_leave: Optional[float], can_move: bool) -> bool:
        """Reveal exactly ONE hidden TRUE mine slot here (count-only rule). Logs to CSV."""
        if self.pos in self.scanned_once:
            return False
        candidates = self._hidden_mines_here()
        if not candidates or self.resource < self.cfg.scan_cost:
            return False

        resource_before = float(self.resource)
        mine_id = int(self.rng.choice(candidates))
        self.resource -= self.cfg.scan_cost

        # Copy TRUE fields into visible for this slot
        tcat = self.env.loc[self.pos, f"TRUE Number {mine_id} mines"]
        trp  = self.env.loc[self.pos, f"TRUE Mine {mine_id} reward_prob"]
        tda  = self.env.loc[self.pos, f"TRUE Mine {mine_id} digs_allowed"]

        self.env.loc[self.pos, f"Number {mine_id} mines"] = tcat
        self.env.loc[self.pos, f"Mine {mine_id} reward_prob"] = trp
        self.env.loc[self.pos, f"Mine {mine_id} digs_allowed"] = tda
        self.env.loc[self.pos, f"Mine {mine_id} digs_remaining"] = tda
        self.env.loc[self.pos, f"Mine {mine_id} revealed"] = True

        self.scanned_once.add(self.pos)

        # Auto-label if fully revealed now (or mark poor if appropriate)
        self._autolabel_if_ready(self.pos)

        # Log & animate
        self._log_step(
            action="scan", decision="scan_here",
            resource_before=resource_before, resource_after=float(self.resource),
            v_stay=v_stay, v_leave=v_leave,
            can_scan_here=True, can_move=can_move,
            mine_id=mine_id, revealed_cat=tcat
        )
        self._snapshot("scan", "scan_here")
        return True

    def _move(self, v_stay: Optional[float], v_leave: Optional[float], can_scan_here: bool) -> bool:
        """Move to a neighbor with count-only preference. Logs to CSV."""
        neigh = self._neighbors()
        if not neigh or self.resource < self.cfg.move_cost:
            return False

        # Prefer neighbors with MORE hidden TRUE mine slots
        eps = 1e-6
        weights = []
        for n in neigh:
            hidden = sum(
                (self.env.loc[n, f"TRUE Number {i} mines"] is not None) and (not bool(self.env.loc[n, f"Mine {i} revealed"]))
                for i in (1, 2, 3)
            )
            weights.append(eps + hidden)
        probs = np.array(weights, dtype=float)
        probs /= probs.sum()

        choice_idx = int(self.rng.choice(len(neigh), p=probs))
        choice = neigh[choice_idx]

        # Pay & move
        resource_before = float(self.resource)
        self.resource -= self.cfg.move_cost
        if (self.pos == self.base) and (choice != self.base):
            self.left_base_once = True
        self.pos = choice
        self.total_moves += 1

        # Open tile for forager and attempt autolabel (e.g., visited empty -> 'poor')
        self.env.loc[self.pos, 'visited_by_explorer'] = True
        self._autolabel_if_ready(self.pos)

        # Log & animate
        self._log_step(
            action="move", decision="leave",
            resource_before=resource_before, resource_after=float(self.resource),
            v_stay=v_stay, v_leave=v_leave,
            can_scan_here=can_scan_here, can_move=True,
            move_probs=list(np.round(probs, 4))
        )
        self._snapshot("move", "leave")
        return True

    # ---- decision values (count-only) ----
    def _values(self) -> Tuple[float, float]:
        """
        A simple, dimensionless pressure comparison based solely on counts:
          - Local pressure ~ (#hidden here)
          - Global pressure ~ (total #hidden elsewhere)
        Costs only gate feasibility; they don't enter the preference directly.
        """
        R_local = len(self._hidden_mines_here())           # 0..3
        R_global = self._global_hidden_count()             # 0..(3*grid)
        stay = self.cfg.beta_local * float(R_local)-self.cfg.scan_cost*self.cfg.gamma
        leave = self.cfg.beta_global * float(max(R_global - R_local, 0))-self.cfg.move_cost*self.cfg.gamma
        return float(stay), float(leave)

    def step(self):
        # If neither action is affordable, stop this phase
        if (self.resource < self.cfg.scan_cost) and (self.resource < self.cfg.move_cost):
            self._log_step(
                action="halt", decision="insufficient_resources",
                resource_before=float(self.resource), resource_after=float(self.resource),
                v_stay=None, v_leave=None,
                can_scan_here=False, can_move=False
            )
            self._snapshot("halt", "insufficient_resources")
            self.t += 1
            return

        v_stay, v_leave = self._values()
        can_scan_here = (
            (self.resource >= self.cfg.scan_cost)
            and (self.pos not in self.scanned_once)
            and (len(self._hidden_mines_here()) > 0)
        )
        can_move = (self.resource >= self.cfg.move_cost) and (len(self._neighbors()) > 0)

        did_action = False
        # Prefer moving if leave-pressure is higher or scanning is impossible
        if (not can_scan_here) or (v_leave > v_stay):
            did_action = self._move(v_stay, v_leave, can_scan_here)

        # If couldn't move, try single scan here (once per cell)
        if (not did_action) and can_scan_here:
            did_action = self._reveal_one_mine_here(v_stay, v_leave, can_move)

        # If still no action possible -> STOP PHASE (do not drain more resources)
        if not did_action:
            self._log_step(
                action="halt", decision="no_actions_left",
                resource_before=float(self.resource), resource_after=float(self.resource),
                v_stay=v_stay, v_leave=v_leave,
                can_scan_here=can_scan_here, can_move=can_move
            )
            self._snapshot("halt", "no_actions_left")

        self.t += 1

    def total_food_left(self) -> float:
        # Return how much of the explorer's allocated food remains.
        return float(self.resource)

    # --- Transfer the map for forager decisions (same shared DataFrame) ---
    def export_env_for_forager(self) -> pd.DataFrame:
        return self.env

    def run(self, max_steps: int = 300, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run explorer for up to `max_steps`. If `csv_path` is provided, save the step log to CSV.
        Returns the pandas DataFrame of the log.
        """
        for _ in range(max_steps):
            if (self.resource < self.cfg.scan_cost) and (self.resource < self.cfg.move_cost):
                # One final log row (if not already logged this tick)
                self._log_step(
                    action="halt", decision="insufficient_resources",
                    resource_before=float(self.resource), resource_after=float(self.resource),
                    v_stay=None, v_leave=None,
                    can_scan_here=False, can_move=False
                )
                self._snapshot("halt", "insufficient_resources")
                break
            self.step()
            if self.log and self.log[-1].get("action") == "halt":
                break

        df = pd.DataFrame(self.log)
        if csv_path:
            # Always safe-guard against nested lists (e.g., move_probs) by converting to JSON-ish strings
            df_to_save = df.copy()
            if "move_probs" in df_to_save.columns:
                df_to_save["move_probs"] = df_to_save["move_probs"].apply(
                    lambda v: ("" if v is None else repr(v))
                )
            df_to_save.to_csv(csv_path, index=False)
        return df

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ------------------- Helpers assumed from your codebase -------------------
# - ensure_runtime_columns(env_df): adds runtime cols like visited_by_explorer, env_empty, etc.
# - _indexify(env_df): ensures MultiIndex by (row, col)
# - _cell_total_remaining_digs_df(env_df, pos): returns total remaining digs in a cell (int)

# ------------------- Forager (Bayesian stay/leave + softmax neighbor move) + CSV logging -------------------
@dataclass
class MVTConfig:
    # Resources & costs
    init_resource: int = 100
    move_cost: int = 10
    dig_cost: int = 5
    gamma: float = 0.1  # retained (not directly used in Bayesian core)
    reward_amount: float = 1.0
    avoid_base: bool = True

    # Trust / mixing
    beta_trust: float = 0.7         # weight on explorer label vs model-based E[next]
    label_value: Dict[str, float] = None  # label→score for neighbor selection
    env_factor: Dict[str, float] = None   # (kept for compatibility; not used in Bayesian core)
    mine_choice_value: Dict[str, float] = None  # (kept for compatibility; can be used for within-cell mine choice)

    # Temperatures
    stay_leave_temp: float = 0.7     # softmax temperature for stay vs leave
    move_temp: float = 0.7           # softmax temperature for neighbor choice
    cost_sensitive_index: float=0.1

    # -------- Bayesian observer (Bornstein-style) --------
    # K discrete types; fixed means and priors; shared Normal noise variance
    K: int = 3
    MU: List[float] = None           # e.g., [0.2, 0.6, 0.9] * reward_amount
    SIGMA2: float = 0.05             # observation noise variance (Normal likelihood)
    PI: List[float] = None           # type priors, sum to 1 (e.g., [0.33, 0.33, 0.34])

    # Map explorer labels to type indices (None if unrevealed/unknown)
    label_to_type: Dict[str, Optional[int]] = None  # {'poor':0, 'neutral':1, 'rich':2}

    # Baseline for leaving: 'env' uses ∑πμ; 'rate' uses running average λ
    baseline_mode: str = "env"
    ewma_eta: float = 0.1            # EWMA step for λ if baseline_mode == 'rate'

    def __post_init__(self):
        if self.mine_choice_value is None:
            self.mine_choice_value = {'rich': 1.0, 'neutral': 0.6, 'poor': 0.2}
        if self.env_factor is None:
            self.env_factor = {'rich': 0.8, 'neutral': 0.5, 'poor': 0.2}
        if self.label_value is None:
            self.label_value = {'rich': 0.8, 'neutral': 0.5, 'poor': 0.2}
        if self.MU is None:
            # Default scaled by reward_amount (you can set explicitly)
            self.MU = [0.2 * self.reward_amount, 0.6 * self.reward_amount, 0.9 * self.reward_amount]
        if self.PI is None:
            self.PI = [1.0 / self.K] * self.K
        if self.label_to_type is None:
            self.label_to_type = {'poor': 0, 'neutral': 1, 'rich': 2}
        # Normalize
        s = sum(self.PI)
        if s <= 0:
            raise ValueError("PI must contain positive entries.")
        self.PI = [p / s for p in self.PI]


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.array(x, dtype=float) / max(temp, 1e-9)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.maximum(e.sum(), 1e-12)


class MVTAgent:
    """
    Bayesian Forager (CSV-logging version):
      - Movement restricted to env['visited_by_explorer'] == True (opened tiles).
      - Mines are diggable iff digs_remaining > 0.
      - Stay vs leave: SOFTMAX over [Q_stay, Q_leave] where
            Q_stay  = E[r_next | this cell] - dig_cost
            Q_leave = baseline - move_cost, baseline in {'env', 'rate'}
      - Neighbor move: SOFTMAX over beta*label_score + (1-beta)*E[r_next | neighbor].
      - Reward observations per cell are stored in r_obs_map[(r,c)] for Bayesian updating.
      - Revealed types (Explorer Label) bypass inference via label_to_type.
      - Use run(max_steps, csv_path="forager.csv") to dump a step-by-step log to CSV.
    """
    def __init__(self, env_df: pd.DataFrame, cfg: MVTConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed if seed is not None else None)

        self.env = ensure_runtime_columns(env_df)  # same reference
        _indexify(self.env)

        self.resource = cfg.init_resource
        size = int(self.env.index.get_level_values(0).max()) + 1
        self.base = (size // 2, size // 2)
        self.pos = self.base
        self.left_base_once = False

        self.total_reward = 0.0
        self.total_digs = 0
        self.t = 0
        self.log: List[Dict] = []

        # Bayesian memory per cell
        self.r_obs_map: Dict[Tuple[int, int], List[float]] = {}   # observed rewards
        self.post_map: Dict[Tuple[int, int], np.ndarray] = {}     # cached P(z|data)

        # Running average (if using rate baseline)
        self.lambda_rate = 0.0

        # Precompute env baseline: sum_k PI[k] * MU[k]
        self.R_ENV = float(sum(p * m for p, m in zip(self.cfg.PI, self.cfg.MU)))

        self.Nrows = int(self.env.index.get_level_values(0).max()) + 1
        self.Ncols = int(self.env.index.get_level_values(1).max()) + 1

        # Animation buffers
        self.frames_intensity: List[np.ndarray] = []
        self.frames_has_mines: List[np.ndarray] = []
        self.frames_pos: List[Tuple[int, int]] = []
        self.frames_reward: List[float] = []
        self.frames_resource: List[float] = []
        self.frames_action: List[str] = []
        self.frames_decision: List[str] = []

        self._log_enter()
        self._snapshot_grid_state(action_label="start", decision_label="starting")

    # ---------- env helpers ----------
    def _cell_overall(self, pos=None):
        if pos is None: pos = self.pos
        return self.env.loc[pos, 'Overall Richness of this environment']  # unchanged

    def _available_mines(self, pos=None) -> List[int]:
        if pos is None: pos = self.pos
        mines = []
        for i in (1, 2, 3):
            rem = self.env.loc[pos, f'Mine {i} digs_remaining']
            cat = self.env.loc[pos, f'Number {i} mines']
            if pd.notna(rem) and float(rem) > 0 and (cat is not None):
                mines.append(i)
        return mines

    def _neighbors(self):
        r, c = self.pos
        neigh = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        valid = [(rr, cc) for rr, cc in neigh if (rr, cc) in self.env.index]
        if self.cfg.avoid_base and self.left_base_once:
            valid = [p for p in valid if p != self.base]
        # only tiles opened by explorer
        valid = [p for p in valid if bool(self.env.loc[p, 'visited_by_explorer'])]
        return valid

    # ---------- Bayesian core ----------
    def _type_index_from_label(self, label: Optional[str]) -> Optional[int]:
        if label is None or (isinstance(label, float) and pd.isna(label)):
            return None
        return self.cfg.label_to_type.get(str(label).lower(), None)

    def _r_list(self, pos: Tuple[int, int]) -> List[float]:
        return self.r_obs_map.setdefault(pos, [])

    def _loglike_sum_normal(self, r_list: List[float], mu: float, sigma2: float) -> float:
        if not r_list:
            return 0.0
        n = len(r_list)
        diffsq = sum((r - mu) * (r - mu) for r in r_list)
        return -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * (diffsq / sigma2)

    def _posterior_over_types(self, pos: Tuple[int, int]) -> np.ndarray:
        label = self.env.loc[pos, 'Explorer Label']
        k_revealed = self._type_index_from_label(label)
        if k_revealed is not None:
            post = np.zeros(self.cfg.K, dtype=float)
            post[k_revealed] = 1.0
            self.post_map[pos] = post
            return post

        r_list = self._r_list(pos)
        if len(r_list) == 0:
            post = np.array(self.cfg.PI, dtype=float)
            self.post_map[pos] = post
            return post

        logs = []
        for k in range(self.cfg.K):
            L = np.log(self.cfg.PI[k]) + self._loglike_sum_normal(r_list, self.cfg.MU[k], self.cfg.SIGMA2)
            logs.append(L)
        logs = np.array(logs, dtype=float)
        m = logs.max()
        w = np.exp(logs - m)
        post = w / np.maximum(w.sum(), 1e-12)
        self.post_map[pos] = post
        return post

    def _E_next_reward(self, pos: Tuple[int, int]) -> float:
        if len(self._available_mines(pos)) == 0:
            return 0.0
        label = self.env.loc[pos, 'Explorer Label']
        k_revealed = self._type_index_from_label(label)
        if k_revealed is not None:
            return float(self.cfg.MU[k_revealed])
        post = self._posterior_over_types(pos)
        return float(np.dot(post, np.array(self.cfg.MU, dtype=float)))

    # ---------- values ----------
    def _Q_stay(self) -> float:
        E_next = self._E_next_reward(self.pos)
        return E_next - float(self.cfg.dig_cost)*float(self.cfg.cost_sensitive_index)

    def _Q_leave(self) -> float:
        if self.cfg.baseline_mode == "rate":
            baseline = self.lambda_rate
        else:
            baseline = self.R_ENV
        return baseline - float(self.cfg.move_cost)*float(self.cfg.cost_sensitive_index)

    # ---------- rich CSV logger ----------
    def _log_step(
        self,
        *,
        action: str,
        decision: str,
        resource_before: float,
        resource_after: float,
        Q_stay: Optional[float],
        Q_leave: Optional[float],
        stay_prob: Optional[float],
        can_dig: Optional[bool],
        can_move: Optional[bool],
        # action-specific (dig)
        mine_id: Optional[int] = None,
        success: Optional[bool] = None,
        reward_gained: Optional[float] = None,
        depleted: Optional[bool] = None,
        remaining_after: Optional[int] = None,
        # action-specific (move)
        move_choice_label: Optional[str] = None,
        move_probs: Optional[List[float]] = None,
        move_label_scores: Optional[List[float]] = None,
        move_e_next: Optional[List[float]] = None
    ):
        # Posterior at current cell (after any updates already applied for this tick)
        post = self.post_map.get(self.pos, None)
        post_cols = {}
        if post is not None:
            for k in range(self.cfg.K):
                post_cols[f"post_k{k}"] = float(post[k])

        row = {
            # time & pos
            "step": int(self.t),
            "row": int(self.pos[0]),
            "col": int(self.pos[1]),
            "action": action,
            "decision": decision,

            # resources
            "resource_before": float(resource_before),
            "resource_after": float(resource_after),

            # values
            "Q_stay": (None if Q_stay is None else float(Q_stay)),
            "Q_leave": (None if Q_leave is None else float(Q_leave)),
            "stay_prob": (None if stay_prob is None else float(stay_prob)),
            "E_next_here": float(self._E_next_reward(self.pos)),

            # feasibility
            "can_dig": (None if can_dig is None else bool(can_dig)),
            "can_move": (None if can_move is None else bool(can_move)),

            # mined counts at current cell
            "available_mines_here": int(len(self._available_mines(self.pos))),

            # labels
            "overall_label": str(self._cell_overall()),
            "explorer_label": str(self.env.loc[self.pos, 'Explorer Label']),

            # dig specifics
            "mine_id": (None if mine_id is None else int(mine_id)),
            "dig_success": (None if success is None else bool(success)),
            "reward_gained": (None if reward_gained is None else float(reward_gained)),
            "depleted": (None if depleted is None else bool(depleted)),
            "remaining_after": (None if remaining_after is None else int(remaining_after)),

            # move specifics (lists converted to repr later when saving)
            "move_choice_label": (None if move_choice_label is None else str(move_choice_label)),
            "move_probs": (None if move_probs is None else list(move_probs)),
            "move_label_scores": (None if move_label_scores is None else list(move_label_scores)),
            "move_e_next": (None if move_e_next is None else list(move_e_next)),

            # running totals
            "reward_total": float(self.total_reward),
            "total_digs": int(self.total_digs),

            # config snapshot
            "cfg_init_resource": int(self.cfg.init_resource),
            "cfg_move_cost": int(self.cfg.move_cost),
            "cfg_dig_cost": int(self.cfg.dig_cost),
            "cfg_reward_amount": float(self.cfg.reward_amount),
            "cfg_beta_trust": float(self.cfg.beta_trust),
            "cfg_stay_leave_temp": float(self.cfg.stay_leave_temp),
            "cfg_move_temp": float(self.cfg.move_temp),
            "cfg_K": int(self.cfg.K),
            "cfg_SIGMA2": float(self.cfg.SIGMA2),
            "cfg_baseline_mode": str(self.cfg.baseline_mode),
            "lambda_rate": float(self.lambda_rate),
            "R_ENV": float(self.R_ENV),
        }
        row.update(post_cols)
        self.log.append(row)

    # ---------- legacy lightweight log (kept for compatibility with your animations) ----------
    def _log(self, **kw):
        # Keep the old lightweight logs for any downstream animation code you already have.
        kw.setdefault('step', self.t)
        kw.setdefault('row', self.pos[0]); kw.setdefault('col', self.pos[1])
        kw.setdefault('resource', self.resource)
        kw.setdefault('reward_total', self.total_reward)
        kw.setdefault('total_digs', self.total_digs)
        kw.setdefault('overall', self._cell_overall())
        kw.setdefault('E_next_here', self._E_next_reward(self.pos))
        post = self.post_map.get(self.pos, None)
        if post is not None:
            for k in range(self.cfg.K):
                kw.setdefault(f'post_k{k}', float(post[k]))
        self.log.append(kw)

    def _log_enter(self):
        # Minimal enter-log for animation compatibility; CSV row is emitted on first decision step
        self._log(action='enter', decision="enter", v_stay=None, v_leave=None, note='entered_cell')

    def _update_cell_empty_flag(self, pos=None):
        if pos is None: pos = self.pos
        has_any = len(self._available_mines(pos)) > 0
        self.env.loc[pos, 'env_empty'] = (not has_any)

    def total_food_left(self) -> float:
        # Return how much of the forager's allocated food remains + reward.
        return float(self.resource + self.total_reward)

    # ---------- depletion snapshots for animation ----------
    def _snapshot_grid_state(self, action_label: str = "", decision_label: str = ""):
        MAX_DIGS_PER_CELL = 18
        inten = np.zeros((self.Nrows, self.Ncols), dtype=float)
        mask  = np.zeros((self.Nrows, self.Ncols), dtype=bool)
        for (r, c) in self.env.index:
            total = _cell_total_remaining_digs_df(self.env, (r, c))
            inten[r, c] = min(total / MAX_DIGS_PER_CELL, 1.0)
            mask[r, c] = (total > 0)

        self.frames_intensity.append(inten)
        self.frames_has_mines.append(mask)
        self.frames_pos.append(tuple(self.pos))
        self.frames_reward.append(float(self.total_reward))
        self.frames_resource.append(float(self.resource))
        self.frames_action.append(action_label if action_label else "")
        self.frames_decision.append(decision_label if decision_label else "")

    def _post_action_snapshot(self, action_label: str, decision_label: str):
        self._snapshot_grid_state(action_label=action_label, decision_label=decision_label)

    # ---------- actions ----------
    def _auto_leave_if_empty(self) -> bool:
        if len(self._available_mines()) > 0:
            return False
        if self.resource < self.cfg.move_cost or len(self._neighbors()) == 0:
            # emit CSV row (halt)
            self._log_step(
                action='halt', decision='stuck_no_mines',
                resource_before=float(self.resource), resource_after=float(self.resource),
                Q_stay=None, Q_leave=None, stay_prob=None,
                can_dig=False, can_move=False
            )
            self._post_action_snapshot("halt", "stuck_no_mines")
            return True
        self._move(auto_leave=True)
        return True

    def _dig(self):
        if self.resource < self.cfg.dig_cost:
            self._log_step(
                action='halt', decision='no_resource_to_dig',
                resource_before=float(self.resource), resource_after=float(self.resource),
                Q_stay=self._Q_stay(), Q_leave=self._Q_leave(), stay_prob=None,
                can_dig=False, can_move=(len(self._neighbors()) > 0)
            )
            self._post_action_snapshot("halt","no_resources_to_dig")
            return False

        mines = self._available_mines()
        if not mines:
            return False

        # Mine choice within the cell (kept compatible with your earlier heuristic)
        vals = []
        for i in mines:
            cat = self.env.loc[self.pos, f'Number {i} mines']
            v = self.cfg.mine_choice_value.get(cat, 0.5)
            vals.append(max(v, 1e-9))
        w = np.array(vals, dtype=float)
        mine_id = int(self.rng.choice(mines, p=w / w.sum()))

        # Dig mechanics
        resource_before = float(self.resource)
        p = self.env.loc[self.pos, f'Mine {mine_id} reward_prob']
        self.resource -= self.cfg.dig_cost
        success = (p is not None) and (self.rng.random() < float(p))
        r_obs = (self.cfg.reward_amount if success else 0.0)

        # Update totals and Bayesian memory
        if success:
            self.total_reward += self.cfg.reward_amount
        self.total_digs += 1
        self._r_list(self.pos).append(float(r_obs))      # store observation for posterior
        self._posterior_over_types(self.pos)             # refresh cached posterior

        # Deplete mine
        rem_col = f"Mine {mine_id} digs_remaining"
        new_rem = int(self.env.loc[self.pos, rem_col]) - 1
        self.env.loc[self.pos, rem_col] = new_rem
        depleted = (new_rem <= 0)
        if depleted:
            self.env.loc[self.pos, f'Number {mine_id} mines'] = None
            self.env.loc[self.pos, f'Mine {mine_id} reward_prob'] = None
            self.env.loc[self.pos, rem_col] = None

        # Update running λ if using rate baseline
        if self.cfg.baseline_mode == "rate":
            g = r_obs - float(self.cfg.dig_cost)
            self.lambda_rate = (1.0 - self.cfg.ewma_eta) * self.lambda_rate + self.cfg.ewma_eta * g

        # Emit CSV row
        self._log_step(
            action='dig', decision='stay',
            resource_before=resource_before, resource_after=float(self.resource),
            Q_stay=self._Q_stay(), Q_leave=self._Q_leave(),
            stay_prob=None,  # only relevant at choice time; here we already chose
            can_dig=True, can_move=(len(self._neighbors()) > 0),
            mine_id=mine_id, success=bool(success), reward_gained=float(r_obs),
            depleted=bool(depleted), remaining_after=(None if depleted else new_rem)
        )
        self._post_action_snapshot("dig","stay")
        return True

    def _move(self, auto_leave: bool = False):
        if self.resource < self.cfg.move_cost:
            self._log_step(
                action='halt', decision='no_resource_to_move',
                resource_before=float(self.resource), resource_after=float(self.resource),
                Q_stay=self._Q_stay(), Q_leave=self._Q_leave(), stay_prob=None,
                can_dig=(len(self._available_mines()) > 0), can_move=False
            )
            self._post_action_snapshot("halt","no_resource_to_move")
            return False

        neigh = self._neighbors()
        if not neigh:
            self._log_step(
                action='halt', decision='no_neighbors',
                resource_before=float(self.resource), resource_after=float(self.resource),
                Q_stay=self._Q_stay(), Q_leave=self._Q_leave(), stay_prob=None,
                can_dig=(len(self._available_mines()) > 0), can_move=False
            )
            self._post_action_snapshot("halt","no_neighbors")
            return False

        # Neighbor softmax: beta * label_score + (1-beta) * E_next(neighbor)
        beta = float(self.cfg.beta_trust)

        labels = [self.env.loc[n, 'Explorer Label'] for n in neigh]
        label_scores = np.array([self.cfg.label_value.get(str(l).lower(), 0.5) if pd.notna(l) else 0.5
                                 for l in labels], dtype=float)

        # Bayesian expected next reward per neighbor
        e_next = np.array([self._E_next_reward(n) for n in neigh], dtype=float)

        # Normalize e_next to a comparable 0–1 scale for mixing (avoid magnitude mismatch)
        scale = max(self.cfg.reward_amount, 1e-6)
        e_scaled = np.clip(e_next / scale, 0.0, 1.0)

        logits = beta * label_scores + (1.0 - beta) * e_scaled
        probs = _softmax(logits, temp=self.cfg.move_temp)

        choice_idx = int(self.rng.choice(len(neigh), p=probs))
        choice = neigh[choice_idx]
        choice_label = labels[choice_idx]

        # Pay and move
        resource_before = float(self.resource)
        self.resource -= self.cfg.move_cost
        if (self.pos == self.base) and (choice != self.base):
            self.left_base_once = True
        self.pos = choice

        # Update running λ if using rate baseline
        if self.cfg.baseline_mode == "rate":
            g = -float(self.cfg.move_cost)
            self.lambda_rate = (1.0 - self.cfg.ewma_eta) * self.lambda_rate + self.cfg.ewma_eta * g

        # Emit CSV row
        self._log_step(
            action='move', decision=('auto_leave' if auto_leave else 'leave'),
            resource_before=resource_before, resource_after=float(self.resource),
            Q_stay=self._Q_stay(), Q_leave=self._Q_leave(),
            stay_prob=None,  # relevant at stay/leave choice time
            can_dig=(len(self._available_mines()) > 0), can_move=True,
            move_choice_label=choice_label,
            move_probs=list(np.round(probs, 4)),
            move_label_scores=list(np.round(label_scores, 3)),
            move_e_next=list(np.round(e_next, 3))
        )
        self._post_action_snapshot("move", "auto_leave" if auto_leave else "leave")
        self._log_enter()
        return True

    # ---------- main loop (softmax stay vs leave) ----------
    def step(self):
        # Auto-leave if no mines here but movement is possible
        if self._auto_leave_if_empty():
            self.t += 1
            return

        can_dig = (self.resource >= self.cfg.dig_cost) and (len(self._available_mines()) > 0)
        can_move = (self.resource >= self.cfg.move_cost) and (len(self._neighbors()) > 0)

        if not can_dig and not can_move:
            self._log_step(
                action='halt', decision='insufficient_actions',
                resource_before=float(self.resource), resource_after=float(self.resource),
                Q_stay=None, Q_leave=None, stay_prob=None,
                can_dig=False, can_move=False
            )
            self._post_action_snapshot("halt","insufficient_actions")
            self.t += 1
            return

        Q_stay, Q_leave = self._Q_stay(), self._Q_leave()
        logits = np.array([
            Q_stay if can_dig else -1e9,
            Q_leave if can_move else -1e9
        ], dtype=float)
        probs = _softmax(logits, temp=self.cfg.stay_leave_temp)
        p_stay = float(probs[0])

        resource_before = float(self.resource)
        choice = int(self.rng.choice([0, 1], p=probs))  # 0=stay(dig), 1=leave(move)

        if choice == 0 and can_dig:
            self._dig()
        elif choice == 1 and can_move:
            self._move(auto_leave=False)
        else:
            # fallback
            if can_dig:
                self._dig()
            else:
                self._move(auto_leave=False)

        # Log a high-level "decision row" for this tick if you want *both* the decision
        # and the action rows. If you prefer only action rows, comment this block out.
        self._log_step(
            action="decide", decision=("stay" if (choice == 0 and can_dig) else "leave"),
            resource_before=resource_before, resource_after=float(self.resource),
            Q_stay=Q_stay, Q_leave=Q_leave, stay_prob=p_stay,
            can_dig=can_dig, can_move=can_move
        )

        self.t += 1

    def run(self, max_steps: int = 300, csv_path: Optional[str] = None) -> pd.DataFrame:
        for _ in range(max_steps):
            # stop if can't do anything
            if (self.resource < self.cfg.dig_cost) and (self.resource < self.cfg.move_cost):
                self._log_step(
                    action='halt', decision='insufficient_resources',
                    resource_before=float(self.resource), resource_after=float(self.resource),
                    Q_stay=None, Q_leave=None, stay_prob=None,
                    can_dig=False, can_move=False
                )
                self._post_action_snapshot("halt", "insufficient_resources")
                break
            self.step()
            # If last *action* row was a hard halt, we can break
            if self.log and isinstance(self.log[-1], dict) and self.log[-1].get('action') == 'halt':
                break

        df = pd.DataFrame(self.log)
        if csv_path:
            # Convert list-like columns to safe strings
            for col in ("move_probs", "move_label_scores", "move_e_next"):
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: ("" if v is None else repr(v)))
            df.to_csv(csv_path, index=False)
        return df


# ------------------- Animation helpers -------------------
def _env_index_or_copy(env: pd.DataFrame) -> pd.DataFrame:
    if list(env.index.names) == ["Row", "Col"]:
        return env
    return env.set_index(["Row", "Col"], drop=False)

def animate_explorer(env: pd.DataFrame, agent: ExplorerAgent, outpath: str = "explore_animation.gif"):
    """
    Colors the grid by TRUE richness (no 'unknown' blocks).
    Dim tiles that still contain any HIDDEN mines (based on per-mine 'revealed' flags).
    """
    frames = len(agent.frames_unrevealed_mask)
    pos_seq = agent.frames_pos
    resource_seq = agent.frames_resource
    action_seq = agent.frames_action
    decision_seq = agent.frames_decision

    env_idx = _env_index_or_copy(env)
    Nrows = int(env_idx.index.get_level_values(0).max()) + 1
    Ncols = int(env_idx.index.get_level_values(1).max()) + 1

    fig, ax = plt.subplots(figsize=(5, 5))
    legend_patches = [mpatches.Patch(color=col, label=lab) for lab, col in RICHNESS_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", title="Richness (TRUE for explorer)")

    tiles = {}
    # Create tiles colored by TRUE richness
    for (r, c), cell in env_idx.iterrows():
        richness_true = cell["TRUE Overall Richness"]
        base_rgb = np.array(mcolors.to_rgb(RICHNESS_COLORS.get(richness_true, "white")))
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=base_rgb, edgecolor="black")
        ax.add_patch(rect)
        tiles[(r, c)] = rect

    # Axes/layout
    ax.set_xlim(-0.5, Ncols - 0.5)
    ax.set_ylim(Nrows - 0.5, -0.5)
    ax.set_xticks(range(Ncols))
    ax.set_yticks(range(Nrows))
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title("Explorer (TRUE map color; dim = has hidden mines)")

    # Agent marker & HUD
    agent_dot, = ax.plot([], [], "bo", markersize=10)
    hud_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left')

    def init_anim():
        agent_dot.set_data([], [])
        hud_text.set_text("")
        return [agent_dot, hud_text] + list(tiles.values())

    def update(frame):
        if frame >= frames:
            frame = frames - 1

        # Keep TRUE richness color each frame
        for (r, c), rect in tiles.items():
            richness_true = env_idx.loc[(r, c), "TRUE Overall Richness"]
            rect.set_facecolor(mcolors.to_rgb(RICHNESS_COLORS.get(richness_true, "white")))

        # Dim tiles that still have any hidden mine slots (from agent snapshot)
        mask = agent.frames_unrevealed_mask[frame]
        for (r, c), rect in tiles.items():
            if mask[r, c]:
                col = np.array(rect.get_facecolor()[:3])
                rect.set_facecolor(np.clip(col * 0.6, 0, 1))

        rr, cc = pos_seq[frame]
        agent_dot.set_data([cc], [rr])
        hud_text.set_text(f"t={frame}  action={action_seq[frame]}\nresource={resource_seq[frame]:.2f}\ndecision={decision_seq[frame]}")
        return [agent_dot, hud_text] + list(tiles.values())

    anim = FuncAnimation(fig, update, init_func=init_anim, frames=frames, interval=150, blit=True, repeat=False)
    anim.save(outpath, writer=PillowWriter(fps=6))
    plt.close(fig)

def animate_forager(env: pd.DataFrame, agent: MVTAgent, outpath: str = "forage_animation.gif"):
    """
    Uses the *live* shared env to recolor each frame from the current visible label
    ('Overall Richness of this environment'), so explorer’s auto-labeling / reveals are shown.
    Dimming reflects remaining digs intensity from the agent’s snapshots.
    """
    frames = len(agent.frames_intensity)
    pos_seq = agent.frames_pos
    reward_seq = agent.frames_reward
    resource_seq = agent.frames_resource
    action_seq = agent.frames_action
    decision_seq = agent.frames_decision

    env_idx = _env_index_or_copy(env)
    Nrows = int(env_idx.index.get_level_values(0).max()) + 1
    Ncols = int(env_idx.index.get_level_values(1).max()) + 1

    fig, ax = plt.subplots(figsize=(5, 5))
    legend_patches = [mpatches.Patch(color=col, label=lab) for lab, col in RICHNESS_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", title="Richness (visible)")

    tiles = {}
    # Create rectangles; we'll recolor every frame from agent.env
    for (r, c), _cell in env_idx.iterrows():
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="white", edgecolor="black")
        ax.add_patch(rect)
        tiles[(r, c)] = rect

    # Axes/layout
    ax.set_xlim(-0.5, Ncols - 0.5)
    ax.set_ylim(Nrows - 0.5, -0.5)
    ax.set_xticks(range(Ncols))
    ax.set_yticks(range(Nrows))
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title("Forager (action & reward)")

    # Agent marker & HUD
    agent_dot, = ax.plot([], [], "ro", markersize=12)
    hud_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left')

    def init_anim():
        agent_dot.set_data([], [])
        hud_text.set_text("")
        return [agent_dot, hud_text] + list(tiles.values())

    def update(frame):
        if frame >= frames:
            frame = frames - 1

        inten = agent.frames_intensity[frame]   # [0..1] remaining digs normalized
        hasm = agent.frames_has_mines[frame]    # bool mines-present mask

        # Recolor from CURRENT visible label in the shared env, then dim by intensity
        for (r, c), rect in tiles.items():
            richness_vis = agent.env.loc[(r, c), "Explorer Label"]
            base_rgb = np.array(mcolors.to_rgb(RICHNESS_COLORS.get(richness_vis, "white")))

            if not hasm[r, c]:
                scale = 0.20  # depleted: keep hue but very dim
            else:
                scale = 0.30 + 0.70 * float(inten[r, c])
            rect.set_facecolor(np.clip(base_rgb * scale, 0, 1))

        rr, cc = pos_seq[frame]
        rr, cc = int(rr), int(cc)
        agent_dot.set_data([cc], [rr])
        hud_text.set_text(
            f"t={frame}  action={action_seq[frame]}\n"
            f"reward_total={reward_seq[frame]:.2f}  resource={resource_seq[frame]:.2f}  decision={decision_seq[frame]}"
        )
        return [agent_dot, hud_text] + list(tiles.values())

    anim = FuncAnimation(fig, update, init_func=init_anim, frames=frames, interval=150, blit=True, repeat=False)
    anim.save(outpath, writer=PillowWriter(fps=6))
    plt.close(fig)

if __name__ == "__main__":
    import pandas as pd

    ROUNDS = 5

    # --- Leader & global settings ---
    ld_cfg = LeaderConfig(total_food=400, alpha=0.3)
    leader = LeaderAgent(ld_cfg, seed=None)

    # ---- Safe helpers for multi-round allocation ----
    def _safe_total_food(ldr):
        if hasattr(ldr, "total_food"):
            return int(getattr(ldr, "total_food"))
        cfg = getattr(ldr, "cfg", None)
        if cfg is not None and hasattr(cfg, "total_food"):
            return int(getattr(cfg, "total_food"))
        raise AttributeError("LeaderAgent has neither 'total_food' nor 'cfg.total_food'.")

    def _safe_forager_share(ldr, default=0.5):
        # Prefer explicit share attr if present
        for name in ("forager_share", "share_forager", "p_forager"):
            if hasattr(ldr, name):
                try:
                    val = float(getattr(ldr, name))
                    return max(0.0, min(1.0, val))
                except Exception:
                    pass
        # Or infer from a (explorer, forager) allocation getter
        if hasattr(ldr, "get_allocation"):
            try:
                alloc = ldr.get_allocation()
                if isinstance(alloc, (list, tuple)) and len(alloc) == 2:
                    e_food, f_food = alloc
                    tot = float(e_food) + float(f_food)
                    if tot > 0:
                        return float(f_food) / tot
            except Exception:
                pass
        return default

    def current_allocation(ldr):
        """
        Determine (explorer_food, forager_food) for THIS round.
        - If leader computed next budgets on the last update, use them.
        - Else compute from total_food * forager_share (or default 50/50).
        """
        # If previous update computed next integers, prefer those
        ef_next = getattr(ldr, "explorer_food_next", None)
        ff_next = getattr(ldr, "forager_food_next", None)
        if isinstance(ef_next, (int, float)) and isinstance(ff_next, (int, float)):
            return int(ef_next), int(ff_next)

        total = _safe_total_food(ldr)
        f_share = _safe_forager_share(ldr, 0.5)
        f_food = int(round(total * f_share))
        e_food = int(total - f_food)
        return e_food, f_food

    # --- Safe replacement for LeaderAgent.update_allocation; then bind it ---
    def _safe_update_allocation(self):
        """
        Safely update the forager/explorer resource split based on forager performance.
        - Multiplicative update on forager_share with bounded change.
        - No division-by-zero; stable when total_reward == 0.
        """
        total_food = getattr(self, "total_food", None)
        if total_food is None:
            total_food = getattr(self.cfg, "total_food", None)
        if total_food is None:
            raise AttributeError("LeaderAgent must have total_food or cfg.total_food.")

        # Ensure persistent share; default 50/50
        if not hasattr(self, "forager_share"):
            self.forager_share = 0.5

        f = getattr(self, "forager", None)
        if f is None:
            # Nothing to update yet
            self.forager_food_next  = int(round(total_food * self.forager_share))
            self.explorer_food_next = int(total_food - self.forager_food_next)
            return

        total_reward = float(getattr(f, "total_reward", 0.0))
        perceived_reward = float(f.total_food_left())  # resource + reward (your method)
        diff = perceived_reward - total_reward

        denom = total_reward if total_reward > 0 else max(abs(perceived_reward), 1.0)
        raw_change = float(self.cfg.alpha) * (diff / denom)

        MAX_DELTA = 0.25  # cap ±25% change/round
        change = max(-MAX_DELTA, min(MAX_DELTA, raw_change))

        new_share = self.forager_share * (1.0 + change)
        self.forager_share = float(max(0.05, min(0.95, new_share)))
        self.explorer_share = 1.0 - self.forager_share

        self.forager_food_next  = int(round(total_food * self.forager_share))
        self.explorer_food_next = int(total_food - self.forager_food_next)

    # Bind the method (monkey patch) BEFORE the loop
    LeaderAgent.update_allocation = _safe_update_allocation

    # --- Per-round summary rows will be appended here and saved at the end ---
    rounds_rows = []

    for r in range(1, ROUNDS + 1):
        print(f"\n========== ROUND {r} ==========")

        # (Re)build env for this round (fresh ground truth & visible map)
        env = init_gridworld(size=3, seed=None)

        # --- Allocation for this round ---
        e_food, f_food = current_allocation(leader)
        print(f"[Leader] Allocation → Explorer: {e_food} | Forager: {f_food}")

        # --- Phase 1: Explorer ---
        ex_cfg = ExplorerConfig(
            init_resource=e_food,
            move_cost=4, scan_cost=2,
            gamma=0.1, beta_local=0.6, beta_global=0.4,
            avoid_base=True, no_backtrack=True
        )
        explorer = ExplorerAgent(env, ex_cfg, seed=None)

        # Save full step log to CSV
        ex_csv = f"explorer_round{r}.csv"
        ex_traj = explorer.run(max_steps=300, csv_path=ex_csv)
        animate_explorer(env, explorer, outpath=f"explore_round{r}.gif")

        # --- Transfer to Forager (same DataFrame reference) ---
        env_for_forager = explorer.export_env_for_forager()

        # --- Phase 2: Forager (Bayesian) ---
        fg_cfg = MVTConfig(
            init_resource=f_food,
            move_cost=4, dig_cost=2,
            reward_amount=1.0,
            # Bayesian params (set to your task)
            K=3,
            MU=[0.2, 0.6, 0.9],   # <- set to design/pilot expected rewards per dig
            PI=[1/3, 1/3, 1/3],
            SIGMA2=0.05,
            label_to_type={'poor':0, 'neutral':1, 'rich':2},
            # policy temps & trust
            beta_trust=0.7, stay_leave_temp=0.2, move_temp=0.2, cost_sensitive_index=0.1,
            # leaving baseline
            baseline_mode="env"   # or "rate"
        )
        forager = MVTAgent(env_for_forager, fg_cfg, seed=None)

        # Save full step log to CSV
        fg_csv = f"forager_round{r}.csv"
        traj = forager.run(max_steps=300, csv_path=fg_csv)
        animate_forager(env_for_forager, forager, outpath=f"forage_round{r}.gif")

        # --- Hand results to Leader and update allocation for NEXT round ---
        # Keep a snapshot of share before update for the summary table
        share_before = _safe_forager_share(leader, 0.5)
        leader.set_forager(forager)
        leader.update_allocation()
        share_after = _safe_forager_share(leader, share_before)

        # --- Round summary (append a row we’ll save later) ---
        row_summary = {
            "round": r,
            # allocations used at start of this round
            "explorer_food_used": int(e_food),
            "forager_food_used": int(f_food),
            "forager_share_before": float(share_before),
            # explorer outcomes
            "explorer_steps": int(len(ex_traj)),
            "explorer_resource_remaining": float(explorer.resource),
            "hidden_mines_left": int(explorer._global_hidden_count()),
            # forager outcomes
            "forager_steps": int(len(traj)),
            "forager_reward_total": float(forager.total_reward),
            "forager_resource_remaining": float(forager.resource),
            "forager_total_digs": int(forager.total_digs),
            # leader plan for next round
            "forager_share_after": float(share_after),
            "next_explorer_food": int(getattr(leader, "explorer_food_next", 0)),
            "next_forager_food": int(getattr(leader, "forager_food_next", 0)),
            # file artifacts
            "explorer_csv": ex_csv,
            "forager_csv": fg_csv,
            "explorer_gif": f"explore_round{r}.gif",
            "forager_gif": f"forage_round{r}.gif",
        }
        rounds_rows.append(row_summary)

        # --- Console summaries (unchanged) ---
        print("=== Exploration Summary ===")
        print(f"Explorer steps: {len(ex_traj)} | Remaining resource: {explorer.resource:.2f} "
              f"| Hidden mines left: {explorer._global_hidden_count()}")
        print("=== Foraging Summary ===")
        print(f"Total reward received: {forager.total_reward:.2f}")
        print(f"Resource remaining: {forager.resource:.2f}")
        print(f"[Leader] Next allocation planned → Explorer: {getattr(leader, 'explorer_food_next', 'n/a')} | "
              f"Forager: {getattr(leader, 'forager_food_next', 'n/a')}")

    # --- Save per-round summary table to CSV at the very end ---
    pd.DataFrame(rounds_rows).to_csv("rounds_summary.csv", index=False)
    print("\nSaved:")
    print(" - rounds_summary.csv")
    for r in range(1, ROUNDS + 1):
        print(f" - explorer_round{r}.csv, forager_round{r}.csv")


# --------------------------- Orchestration Adapter (Explorer + Forager) ---------------------------
def simulate_round(cfg: 'MVTConfig' = None,
                   seed: int = None,
                   env_size: int = 3,
                   explorer_steps: int = 200,
                   forager_steps: int = 300, **env_kwargs):
    """Run ONE explorer+forager round on a fresh grid and return summary metrics.

    Returns a dict with at least:
        - 'overall_reward' : float  (sum of rewards earned by the forager)
        - 'explorer_resource_start' : float
        - 'forager_resource_start'  : float
        - 'explorer_resource_final' : float
        - 'forager_resource_final'  : float

    Notes
    -----
    - Builds a new grid with init_gridworld(size=env_size, seed=seed).
    - ExplorerAgent opens tiles (sets visited_by_explorer=True) for the Forager.
    - Forager (MVTAgent) moves/digs only on opened tiles and accumulates rewards.
    - START resources are captured *immediately after agent creation* and BEFORE any steps.
    """
    # Default configs
    cfg = cfg if cfg is not None else MVTConfig()
    exp_cfg = ExplorerConfig()

    # World
    env = init_gridworld(size=env_size, seed=seed, **env_kwargs)

    # Explorer: create and capture START resource
    explorer = ExplorerAgent(env, exp_cfg, seed=seed)
    explorer_start = float(getattr(explorer, "resource", float(exp_cfg.init_resource)))

    # Let explorer open tiles
    _ = explorer.run(max_steps=explorer_steps, csv_path=None)

    # Forager: create and capture START resource (AFTER explorer phase, BEFORE any forager steps)
    forager = MVTAgent(env, cfg, seed=seed)
    forager_start = float(getattr(forager, "resource", float(cfg.init_resource)))

    # Run forager
    log_df = forager.run(max_steps=forager_steps, csv_path=None)

    # Compute overall reward
    overall = getattr(forager, "total_reward", None)
    if overall is None and log_df is not None and "reward" in log_df.columns:
        overall = float(log_df["reward"].fillna(0).sum())
    overall = float(overall) if overall is not None else 0.0

    return {
        "overall_reward": overall,
        "explorer_resource_start": explorer_start,
        "forager_resource_start": forager_start,
        "explorer_resource_final": float(getattr(explorer, "resource", 0.0)),
        "forager_resource_final": float(getattr(forager, "resource", 0.0)),
    }


# --------------------------- Sequential Session (Persistent Memory) ---------------------------
def simulate_session(cfg: 'MVTConfig' = None,
                     seed: int = None,
                     env_size: int = 3,
                     explorer_steps: int = 200,
                     forager_steps: int = 300,
                     n_rounds: int = 10,
                     reset_resources_each_round: bool = False):
    """Run a multi-round session where the same Explorer/Forager and env persist.

    Memory (e.g., forager.r_obs_map/post_map, explorer visited tiles, remaining digs) 
    is preserved across rounds. You can optionally reset agent resources each round.

    Returns: list of dict rows (one per round) with:
        - 'round'
        - 'overall_reward'      : reward EARNED in THAT round (not cumulative)
        - 'explorer_resource_start', 'forager_resource_start'
    """
    cfg = cfg if cfg is not None else MVTConfig()
    exp_cfg = ExplorerConfig()

    # World and agents persist across all rounds
    env = init_gridworld(size=env_size, seed=seed, **env_kwargs)
    explorer = ExplorerAgent(env, exp_cfg, seed=seed)
    forager = MVTAgent(env, cfg, seed=seed)

    rows = []
    for r in range(1, int(n_rounds) + 1):
        # Optionally reset only RESOURCES; keep memory and env as-is
        if reset_resources_each_round:
            explorer.resource = float(exp_cfg.init_resource)
            forager.resource = float(cfg.init_resource)

        # Snapshot starting resources
        exp_start = float(getattr(explorer, "resource", float(exp_cfg.init_resource)))
        for_start = float(getattr(forager, "resource", float(cfg.init_resource)))

        # Explorer opens more tiles
        _ = explorer.run(max_steps=explorer_steps, csv_path=None)

        # Forager collects reward; compute this round's delta
        prev_total = float(getattr(forager, "total_reward", 0.0) or 0.0)
        _ = forager.run(max_steps=forager_steps, csv_path=None)
        new_total = float(getattr(forager, "total_reward", 0.0) or 0.0)
        round_reward = new_total - prev_total

        rows.append({
            "round": r,
            "overall_reward": round_reward,
            "explorer_resource_start": exp_start,
            "forager_resource_start": for_start,
        })

    return rows




# --------------------------- Sequential Session (Persistent Memory) ---------------------------
def simulate_session(cfg: 'MVTConfig' = None,
                     seed: int = None,
                     env_size: int = 3,
                     explorer_steps: int = 200,
                     forager_steps: int = 300,
                     n_rounds: int = 10,
                     reset_resources_each_round: bool = False,
                     explorer_reset_mode: str = "none",   # 'none' | 'set_to_init' | 'topup_to_init'
                     forager_reset_mode: str = "none"     # 'none' | 'set_to_init' | 'topup_to_init'
                     ):
    """Run a multi-round session with persistent env/agents and optional per-round resource resets.

    Memory (forager posteriors, explorer revealed tiles, remaining digs, etc.) persists.
    You can reset agent resources each round via modes:
      - 'none'         : never reset; resources carry over and may reach 0
      - 'set_to_init'  : set resource to its init value each round
      - 'topup_to_init': raise resource to at least its init value (no decrease)

    The legacy boolean `reset_resources_each_round=True` is treated as 'set_to_init' for both agents.

    Returns: list of dicts (one per round) with fields:
      - round
      - overall_reward               (reward earned *in that round*)
      - explorer_resource_start
      - forager_resource_start
    """
    cfg = cfg if cfg is not None else MVTConfig()
    exp_cfg = ExplorerConfig()

    # World and agents persist
    env = init_gridworld(size=env_size, seed=seed, **env_kwargs)
    explorer = ExplorerAgent(env, exp_cfg, seed=seed)
    forager = MVTAgent(env, cfg, seed=seed)

    # Back-compat: old boolean means set_to_init for both
    if reset_resources_each_round:
        explorer_reset_mode = "set_to_init"
        forager_reset_mode = "set_to_init"

    def _apply_reset(agent, mode: str, init_val: float):
        if mode == "set_to_init":
            agent.resource = float(init_val)
        elif mode == "topup_to_init":
            agent.resource = float(max(float(getattr(agent, "resource", 0.0)), float(init_val)))
        # else 'none': do nothing

    rows = []
    for r in range(1, int(n_rounds) + 1):
        # Reset per chosen modes (before logging start)
        _apply_reset(explorer, explorer_reset_mode, exp_cfg.init_resource)
        _apply_reset(forager,  forager_reset_mode,  cfg.init_resource)

        # Snapshot start resources
        exp_start = float(getattr(explorer, "resource", float(exp_cfg.init_resource)))
        for_start = float(getattr(forager,  "resource", float(cfg.init_resource)))

        # Explorer explores more
        _ = explorer.run(max_steps=explorer_steps, csv_path=None)

        # Forager collects reward; compute this round's delta
        prev_total = float(getattr(forager, "total_reward", 0.0) or 0.0)
        _ = forager.run(max_steps=forager_steps, csv_path=None)
        new_total = float(getattr(forager, "total_reward", 0.0) or 0.0)
        round_reward = new_total - prev_total

        rows.append({
            "round": r,
            "overall_reward": round_reward,
            "explorer_resource_start": exp_start,
            "forager_resource_start": for_start,
        })

    return rows


# --------------------------- Session with Fresh Env Per Round ---------------------------
def simulate_session_fresh_env(cfg_params: dict = None,
                               exp_params: dict = None,
                               seed: int = None,
                               env_size: int = 3,
                               explorer_steps: int = 200,
                               forager_steps: int = 300,
                               n_rounds: int = 10,
                               start_resource_explorer: float = 200.0,
                               start_resource_forager: float = 200.0, **env_kwargs):
    """Run a multi-round session where EACH ROUND uses a NEW environment and NEW agents.

    - All unused resources are cleared at end of each round.
    - Each round starts explorer/forager with the provided start resources (default 200).
    - Returns list[dict] with per-round summary:
        {'round', 'overall_reward', 'explorer_resource_start', 'forager_resource_start'}

    Parameters
    ----------
    cfg_params : dict
        Values to construct MVTConfig(**cfg_params). Any missing fields use defaults.
        `init_resource` will be overridden to `start_resource_forager` for the run.
    exp_params : dict
        Values to construct ExplorerConfig(**exp_params). Any missing fields use defaults.
        `init_resource` will be overridden to `start_resource_explorer` for the run.
    """
    cfg_params = dict(cfg_params or {})
    exp_params = dict(exp_params or {})

    rows = []
    rng_seed = seed

    for r in range(1, int(n_rounds) + 1):
        # Rebuild world, configs, agents fresh each round
        # (Override init_resource to enforce start resources = 200 by default)
        from dataclasses import replace, is_dataclass  # local import to avoid leaks

        # Build configs; override init resources
        tmp_cfg = MVTConfig(**cfg_params) if cfg_params else MVTConfig()
        tmp_cfg.init_resource = float(start_resource_forager)

        tmp_exp = ExplorerConfig(**exp_params) if exp_params else ExplorerConfig()
        tmp_exp.init_resource = float(start_resource_explorer)

        # Fresh environment with per-round seed if given
        env = init_gridworld(size=env_size, seed=(None if rng_seed is None else rng_seed + r), **env_kwargs)

        explorer = ExplorerAgent(env, tmp_exp, seed=(None if rng_seed is None else rng_seed + 1000 + r))
        # Snapshot start resources
        exp_start = float(getattr(explorer, "resource", tmp_exp.init_resource))

        # Explorer opens tiles
        _ = explorer.run(max_steps=explorer_steps, csv_path=None)

        forager = MVTAgent(env, tmp_cfg, seed=(None if rng_seed is None else rng_seed + 2000 + r))
        for_start = float(getattr(forager, "resource", tmp_cfg.init_resource))

        # Forager collects reward (round-local)
        prev_total = float(getattr(forager, "total_reward", 0.0) or 0.0)
        _ = forager.run(max_steps=forager_steps, csv_path=None)
        new_total = float(getattr(forager, "total_reward", 0.0) or 0.0)
        round_reward = new_total - prev_total

        rows.append({
            "round": r,
            "overall_reward": round_reward,
            "explorer_resource_start": exp_start,
            "forager_resource_start": for_start,
        })

    return rows

