import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel

class AllocateModel(optGrbModel):
    def __init__(self, uncertainty, uncertainty_quantile=0.5, pred_len=88):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_quantile = uncertainty_quantile
        self.uncertainty_bar = np.quantile(self.uncertainty, self.uncertainty_quantile)
        self.pred_len = pred_len 
        super().__init__()

    def update_uncertainty(self, uncertainty):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_quantile = np.quantile(self.uncertainty, self.uncertainty_quantile)
        # print(self.uncertainty_bar)

    def _getModel(self):
        # create a model
        optmodel = gp.Model()

        # TOTAL number of steps is always the max (88)
        total_prediction_steps = len(self.uncertainty[0]) 

        # variables
        action = optmodel.addVars(total_prediction_steps, name="action", vtype=GRB.CONTINUOUS)

        # model sense
        optmodel.ModelSense = GRB.MINIMIZE

        # constraints
        # This is your new "shrinking horizon" logic!
        # We only sum up to the *actual* remaining steps (self.pred_len)
        optmodel.addConstr(gp.quicksum(self.uncertainty[0,i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1)

        # Force all steps *after* our horizon to be 0
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
        # create a model
        optmodel = gp.Model()
        # variables
        action = optmodel.addVars(self.pred_len, name="action", vtype=GRB.CONTINUOUS)
        # model sense
        optmodel.ModelSense = GRB.MINIMIZE
        # constraints
        optmodel.addConstr(gp.quicksum(self.uncertainty[0,i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1)
        return optmodel, action
