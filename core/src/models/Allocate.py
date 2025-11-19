# models/Allocate.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel

class AllocateModel(optGrbModel):
    """
    Deterministic LP with optional cap on the *first* step of the plan:
      sum_i action[i] == 1                  (fractions of remaining budget)
      dot(uncertainty[0], action[:H]) <= r  (your conformal constraint)
      0 <= action[i]                        (implicitly by var type)
      action[0] <= first_step_cap           (optional, only if provided)

    Notes
    -----
    * We only cap action[0] (the "buy now" fraction). This is re-applied each MPC
      tick as the horizon shrinks, so you get a per-step spending cap.
    * On the last step (pred_len == 1) we DO NOT cap to avoid infeasibility.
    * This does NOT change PNO unless you pass a cap there too.
    """

    def __init__(
        self,
        uncertainty,
        uncertainty_quantile=0.5,
        pred_len=88,
        *,
        first_step_cap: float | None = None,  # NEW: max fraction you can invest *now* (0..1)
        seed: int = 42,
        threads: int = 1,
        method: int = 1,        # 1 = Dual Simplex
        crossover: int = 0,     # keep deterministic if Barrier were used
        numeric_focus: int | None = None,  # 0..3
        quiet: bool = True
    ):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_quantile = float(uncertainty_quantile)
        self.uncertainty_bar = np.quantile(self.uncertainty, self.uncertainty_quantile)
        self.pred_len = int(pred_len)

        # NEW
        self.first_step_cap = None
        if first_step_cap is not None:
            # clamp to [0, 1]
            self.first_step_cap = float(max(0.0, min(1.0, first_step_cap)))

        # deterministic knobs
        self.seed = int(seed)
        self.threads = int(threads)
        self.method = int(method)
        self.crossover = int(crossover)
        self.numeric_focus = numeric_focus
        self.quiet = quiet

        super().__init__()

    def update_uncertainty(self, uncertainty):
        self.uncertainty = np.array(uncertainty)
        # recompute the *bar*
        self.uncertainty_bar = np.quantile(self.uncertainty, self.uncertainty_quantile)

    def _getModel(self):
        optmodel = gp.Model()

        # Silence output if desired
        if self.quiet:
            optmodel.Params.OutputFlag = 0

        # Deterministic settings
        optmodel.Params.Threads = self.threads
        optmodel.Params.Seed = self.seed
        optmodel.Params.Method = self.method
        optmodel.Params.Crossover = self.crossover
        if self.numeric_focus is not None:
            optmodel.Params.NumericFocus = int(self.numeric_focus)

        total_prediction_steps = len(self.uncertainty[0])

        # variables
        action = optmodel.addVars(total_prediction_steps, name="action", vtype=GRB.CONTINUOUS, lb=0.0)

        # model sense
        optmodel.ModelSense = GRB.MINIMIZE

        # constraints (shrinking horizon on the first pred_len entries)
        optmodel.addConstr(
            gp.quicksum(self.uncertainty[0, i] * action[i] for i in range(self.pred_len))
            <= self.uncertainty_bar,
            name="uncertainty_budget"
        )
        optmodel.addConstr(
            gp.quicksum(action[i] for i in range(self.pred_len)) == 1.0,
            name="sum_to_one"
        )

        # NEW: cap the *first* step only (the action you’ll take now).
        # IMPORTANT: do not cap when pred_len == 1, or the model would be infeasible.
        if self.first_step_cap is not None and self.pred_len > 1:
            optmodel.addConstr(action[0] <= self.first_step_cap, name="first_step_cap")

        # force actions beyond our horizon to 0
        if self.pred_len < total_prediction_steps:
            for i in range(self.pred_len, total_prediction_steps):
                optmodel.addConstr(action[i] == 0.0, name=f"beyond_horizon_{i}")

        return optmodel, action


class AllocateModelOld(optGrbModel):
    def __init__(self, uncertainty, uncertainty_bar):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_bar = uncertainty_bar
        self.pred_len = len(self.uncertainty[0])
        super().__init__()

    def _getModel(self):
        optmodel = gp.Model()
        action = optmodel.addVars(self.pred_len, name="action", vtype=GRB.CONTINUOUS, lb=0.0)
        optmodel.ModelSense = GRB.MINIMIZE
        optmodel.addConstr(gp.quicksum(self.uncertainty[0, i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1.0)
        return optmodel, action
