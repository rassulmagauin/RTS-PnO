# models/Allocate.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel

class AllocateModel(optGrbModel):
    """
    Deterministic LP:
      - Single-thread (Threads=1)
      - Fixed Seed
      - Fixed Method (dual simplex by default)
      - Optional: disable crossover if Barrier is used (kept 0 for safety, harmless for simplex)
    """
    def __init__(
        self,
        uncertainty,
        uncertainty_quantile=0.5,
        pred_len=88,
        *,
        seed: int = 42,
        threads: int = 1,
        method: int = 1,        # 0=Primal, 1=Dual (recommended), 2=Barrier, 4=Deterministic concurrent
        crossover: int = 0,     # if Barrier is ever used, keep deterministic
        numeric_focus: int | None = None,  # 0..3, leave None unless you see numeric instability
        quiet: bool = True
    ):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_quantile = float(uncertainty_quantile)
        self.uncertainty_bar = np.quantile(self.uncertainty, self.uncertainty_quantile)
        self.pred_len = int(pred_len)

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
        # BUGFIX: we must recompute the *bar*, not overwrite the quantile value
        self.uncertainty_bar = np.quantile(self.uncertainty, self.uncertainty_quantile)

    def _getModel(self):
        # create model
        optmodel = gp.Model()

        # Silence output if desired
        if self.quiet:
            optmodel.Params.OutputFlag = 0

        # Deterministic settings
        optmodel.Params.Threads = self.threads          # single-thread
        optmodel.Params.Seed = self.seed                # fixed seed
        optmodel.Params.Method = self.method            # e.g., 1 = Dual Simplex
        optmodel.Params.Crossover = self.crossover      # keep deterministic if Barrier is used
        if self.numeric_focus is not None:
            optmodel.Params.NumericFocus = int(self.numeric_focus)

        # TOTAL number of steps in the uncertainty array (usually 88)
        total_prediction_steps = len(self.uncertainty[0])

        # variables
        action = optmodel.addVars(total_prediction_steps, name="action", vtype=GRB.CONTINUOUS)

        # model sense
        optmodel.ModelSense = GRB.MINIMIZE

        # constraints (shrinking horizon)
        optmodel.addConstr(gp.quicksum(self.uncertainty[0, i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1)

        # force actions beyond our horizon to 0
        if self.pred_len < total_prediction_steps:
            for i in range(self.pred_len, total_prediction_steps):
                optmodel.addConstr(action[i] == 0)

        return optmodel, action


class AllocateModelOld(optGrbModel):
    def __init__(self, uncertainty, uncertainty_bar):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_bar = uncertainty_bar
        self.pred_len = len(self.uncertainty[0])
        super().__init__()

    def _getModel(self):
        optmodel = gp.Model()
        action = optmodel.addVars(self.pred_len, name="action", vtype=GRB.CONTINUOUS)
        optmodel.ModelSense = GRB.MINIMIZE
        optmodel.addConstr(gp.quicksum(self.uncertainty[0, i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1)
        return optmodel, action
