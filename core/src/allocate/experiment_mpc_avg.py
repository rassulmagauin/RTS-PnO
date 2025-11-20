import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.case_logger import CaseLogger
import gurobipy as gp  # NEW

import models
from models.Allocate import AllocateModel
from common.experiment import Experiment
from allocate.data_provider import get_allocating_loader_and_dataset

class MpcAvgExperiment(Experiment):
    """
    Averaged MPC experiment over all test windows, using the PnO-trained forecaster.
    """
    def __init__(self, configs):
        super().__init__(configs)
        print("ok, setting up the mpc_avg test...")

        # 0) make Gurobi deterministic (applies to every model created afterwards)
        self._configure_gurobi()  # NEW

        # 1) load the pno model (the 'brain') that we already trained
        self.prev_exp_dir = os.path.join('output', configs.prev_exp_id)
        self._build_forecast_model()

        # 2) load the constraint.pt file (the 'r' vector)
        self._load_constraint()

        # 3) build the gurobi thingy
        self._build_allocate_model()

        # 4) test loader (fixed ordering; no shuffle)
        self.test_loader, self.test_set = get_allocating_loader_and_dataset(
            self.configs, self.allocate_model, split='test', shuffle=False
        )
        self.scaler = self.test_set.scaler
        print("...all set up.")

    # NEW: one place to set deterministic Gurobi params
    def _configure_gurobi(self):
        gp.setParam("OutputFlag", int(getattr(self.configs, "grb_output", 0)))
        gp.setParam("Threads",    int(getattr(self.configs, "grb_threads", 1)))
        gp.setParam("Method",     int(getattr(self.configs, "grb_method", 1)))     # dual simplex
        gp.setParam("Crossover",  int(getattr(self.configs, "grb_crossover", 0))) # no crossover
        gp.setParam("Seed",       int(getattr(self.configs, "grb_seed", 42)))
        # (Optional but helpful) keep numerics stable:
        if hasattr(self.configs, "grb_numeric_focus"):
            gp.setParam("NumericFocus", int(self.configs.grb_numeric_focus))

    def _build_forecast_model(self):
        self.forecast_model = getattr(models, self.configs.model)(self.configs)
        if self.configs.use_multi_gpu:
            self.forecast_model = nn.DataParallel(self.forecast_model, device_ids=self.configs.gpus)
        self.forecast_model.to(self.device)
        self.model = self.forecast_model

        if self.configs.load_prev_weights:
            model_path = os.path.join(self.prev_exp_dir, 'model.pt')
            print(f"loading model from: {model_path}")
            self.load_checkpoint(model_path=model_path)

        self.forecast_model.eval()  # ensure dropout/BN are deterministic

    def _load_constraint(self):
        constraint_path = os.path.join(self.prev_exp_dir, 'constraint.pt')
        print(f"loading constraint file: {constraint_path}")
        self.constraint = torch.load(constraint_path, map_location='cpu')

    def _build_allocate_model(self):
        H = self.configs.pred_len
        self.allocate_model = AllocateModel(
            self.constraint.numpy(),
            self.configs.uncertainty_quantile,
            pred_len=H,
            first_step_cap=getattr(self.configs, "mpc_first_step_cap", None),
        )
        print("Gurobi allocation model built.")

    @torch.no_grad()
    def evaluate(self):
        print("ok running the big evaluation loop...")
        cases_dir = os.path.join(self.exp_dir, "cases_test")
        os.makedirs(cases_dir, exist_ok=True)
        mpc_jsonl_path = os.path.join(cases_dir, "cases.jsonl")
        case_logger = CaseLogger(mpc_jsonl_path)

        all_my_regrets = []
        case_idx = 0

        for batch in tqdm(self.test_loader, desc="Evaluating MPC on Test Set"):
            batch_x_scaled, batch_y_scaled, _, _ = batch
            batch_x_scaled = batch_x_scaled.cpu().numpy()
            batch_y_scaled = batch_y_scaled.cpu().numpy()
            batch_y_unscaled = self.scaler.inverse_transform(batch_y_scaled)

            for j in range(batch_x_scaled.shape[0]):
                problem_x_scaled = batch_x_scaled[j]
                problem_y_unscaled = batch_y_unscaled[j]

                regret, preds_1step, alloc_abs, alloc_frac, real_future = \
                    self._run_one_simulation(
                        problem_x_scaled,
                        problem_y_unscaled,
                        return_details=True
                    )

                all_my_regrets.append(regret)
                mse_i = float(np.mean((preds_1step - real_future) ** 2))
                mae_i = float(np.mean(np.abs(preds_1step - real_future)))
                optimal_cost = float(np.min(real_future))
                alloc_cost = float(np.dot(alloc_abs, real_future))
                rel_regret = float(regret) / optimal_cost if optimal_cost != 0 else None

                # unified schema
                case_logger.log(
                    split="test",
                    idx=int(case_idx),
                    case_id=int(case_idx),
                    algo="mpc",
                    regret=float(regret),
                    rel_regret=rel_regret,
                    true_prices=real_future.tolist(),
                    pred_prices=preds_1step.tolist(),
                    alloc=alloc_abs.tolist(),
                    optimal_cost=optimal_cost,
                    alloc_cost=alloc_cost,
                    mse=mse_i,
                    mae=mae_i,
                    extra={"alloc_frac_remaining": alloc_frac.tolist()},
                )
                case_idx += 1

        case_logger.close()
        avg_regret = float(np.mean(all_my_regrets))
        print("\n--- Averaged MPC Evaluation Complete ---")
        print(f"Total Problems Simulated: {len(all_my_regrets)}")
        print(f"Average MPC Regret:       {avg_regret:.8f}")

        metrics = {'Avg_MPC_Regret': avg_regret}
        self._save_results(metrics)
        return metrics

    def _run_one_simulation(self, history_data_scaled, future_data_unscaled, return_details=False):
        H = self.configs.pred_len
        history_scaled_2d = history_data_scaled.reshape(-1, 1)
        current_history_unscaled = list(self.scaler.inverse_transform(history_scaled_2d).squeeze())
        true_future_y = future_data_unscaled.astype(float)

        per_step_pred_1ahead, per_step_alloc_abs, per_step_alloc_frac = [], [], []
        I_rem, total_cost_paid = 1.0, 0.0

        for t in range(H):
            history_2d_unscaled = np.array(current_history_unscaled).reshape(-1, 1)
            history_scaled = self.scaler.transform(history_2d_unscaled)
            history_tensor = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

            pred_y_scaled = self.forecast_model(history_tensor)
            pred_vec = pred_y_scaled.cpu().numpy().ravel()              # (H,)
            pred_y_unscaled = self.scaler.inverse_transform(
                pred_vec.reshape(-1, 1)                                 # (H,1)
            ).ravel()                                                   # (H,)
            pred_1ahead = float(pred_y_unscaled[0])
            per_step_pred_1ahead.append(pred_1ahead)

            remaining_steps = H - t
            current_opt_model = AllocateModel(
                self.constraint.numpy(),
                self.configs.uncertainty_quantile,
                pred_len=remaining_steps,
                first_step_cap=getattr(self.configs, "mpc_first_step_cap", None),  # NEW
            )
            current_opt_model.setObj(pred_y_unscaled)
            sol, _ = current_opt_model.solve()
            a_star = float(sol[0])

            if t == H - 1:
                amount_to_buy = float(I_rem)
            else:
                a_star = float(min(max(a_star, 0.0), 1.0))
                amount_to_buy = float(a_star * I_rem)

            per_step_alloc_abs.append(amount_to_buy)
            per_step_alloc_frac.append(a_star if t < H - 1 else 1.0)

            true_price_this_step = float(true_future_y[t])
            total_cost_paid += amount_to_buy * true_price_this_step
            I_rem = max(I_rem - amount_to_buy, 0.0)

            current_history_unscaled.pop(0)
            current_history_unscaled.append(true_future_y[t])

        total_bought = 1.0 - I_rem
        avg_cost_paid = 0.0 if total_bought < 1e-6 else total_cost_paid / total_bought
        regret = float(avg_cost_paid - float(np.min(true_future_y)))

        if return_details:
            return (
                regret,
                np.asarray(per_step_pred_1ahead, dtype=float),
                np.asarray(per_step_alloc_abs, dtype=float),
                np.asarray(per_step_alloc_frac, dtype=float),
                np.asarray(true_future_y, dtype=float),
            )
        return regret

    def _save_results(self, metrics):
        res_path = os.path.join(self.exp_dir, 'mpc_avg_result.json')
        with open(res_path, 'w') as fout:
            json.dump({k: float(v) for k, v in metrics.items()}, fout, indent=4)
        print(f"Averaged MPC results saved to {res_path}")
