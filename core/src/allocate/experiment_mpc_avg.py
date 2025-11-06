import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import models
from models.Allocate import AllocateModel
from common.experiment import Experiment
from allocate.data_provider import get_allocating_loader_and_dataset

class MpcAvgExperiment(Experiment):
    """
    This is class for the averaged MPC experiment.
    
    My idea: run a mini-MPC simulation on EACH of the 4,506 test(for usdcny) windows
    and then average the regret so I can compare it to the pno result.
    """
    def __init__(self, configs):
        super().__init__(configs)
        print("ok, setting up the mpc_avg test...")
        
        # 1. load the pno model (the 'brain') that we already trained
        self.prev_exp_dir = os.path.join('output', configs.prev_exp_id)
        self._build_forecast_model() 
        
        # 2. load the constraint.pt file (the 'r' vector)
        self._load_constraint()

        # 3. build the gurobi thingy
        self._build_allocate_model()

        # 4. load the test problems... this is the loader from allocate_data_provider
        # it gives us all 4506 test "roads"
        self.test_loader, self.test_set = get_allocating_loader_and_dataset(
            self.configs, 
            self.allocate_model, 
            split='test', 
            shuffle=False
        )
        # get the scaler so we can un-scale prices
        self.scaler = self.test_set.scaler
        
        print("...all set up.")

    def _build_forecast_model(self):
        # this just loads the PatchTST model
        self.forecast_model = getattr(models, self.configs.model)(self.configs)
        if self.configs.use_multi_gpu:
            self.forecast_model = nn.DataParallel(self.forecast_model, device_ids=self.configs.gpus)
        self.forecast_model.to(self.device)
        self.model = self.forecast_model # for compatibility
        
        # make sure to load the weights from the pno experiment!!
        if self.configs.load_prev_weights:
            model_path = os.path.join(self.prev_exp_dir, 'model.pt')
            print(f"loading model from: {model_path}")
            self.load_checkpoint(model_path=model_path)
            
        self.forecast_model.eval() # MUST set this or it will dropout/etc

    def _load_constraint(self):
        # this is the 'r' vector we saved
        constraint_path = os.path.join(self.prev_exp_dir, 'constraint.pt')
        print(f"loading constraint file: {constraint_path}")
        self.constraint = torch.load(constraint_path, map_location='cpu')

    def _build_allocate_model(self):
        H = self.configs.pred_len 
        self.allocate_model = AllocateModel(
            self.constraint.numpy(), 
            self.configs.uncertainty_quantile,
            pred_len=H 
        )
        print("Gurobi allocation model built.")

    @torch.no_grad()
    def evaluate(self):
        # "outer loop" that just manages the 4506 test problems.
        
        print("ok running the big evaluation loop...")
        
        all_my_regrets = []

        # this is the main loop, goes thru all 4506 test problems
        # the batch size is like 32, so this loops ~141 times
        for batch in tqdm(self.test_loader, desc="Evaluating MPC on Test Set"):
            # batch_x is scaled history, batch_y is scaled future
            batch_x_scaled, batch_y_scaled, _, _ = batch
            
            batch_x_scaled = batch_x_scaled.cpu().numpy() # shape (32, 440)
            batch_y_scaled = batch_y_scaled.cpu().numpy() # shape (32, 88)
            
            # un-scale the 'true future' prices
            batch_y_unscaled = self.scaler.inverse_transform(batch_y_scaled)
            
            # now loop over each problem in the batch (e.g., 32 times)
            for j in range(batch_x_scaled.shape[0]):
                
                # get the history and future for *this one* problem
                problem_x_scaled = batch_x_scaled[j]
                problem_y_unscaled = batch_y_unscaled[j]
                
                # run the simulation for this one road
                # this helper function does all the hard work
                regret = self._run_one_simulation(problem_x_scaled, problem_y_unscaled)
                
                all_my_regrets.append(regret)

        # --- All 4,506 problems are done ---
        average_mpc_regret = np.mean(all_my_regrets)

        print("\n--- Averaged MPC Evaluation Complete ---")
        print(f"Total Problems Simulated: {len(all_my_regrets)}")
        print(f"Average MPC Regret:       {average_mpc_regret:.8f}")
        
        metrics = {
            'Avg_MPC_Regret': average_mpc_regret,
        }
        
        self._save_results(metrics)
        return metrics

    def _run_one_simulation(self, history_data_scaled, future_data_unscaled):
        # This function runs one full 88-step mpc sim

        
        H = self.configs.pred_len # this is 88
        
        # history_data_scaled is (440,) need (440, 1)
        history_scaled_2d = history_data_scaled.reshape(-1, 1) 
        # get the unscaled history to start our simulation
        current_history_unscaled = list(self.scaler.inverse_transform(history_scaled_2d).squeeze()) # .squeeze() to make it 1D
        
        # this is the "answer key" for this road
        true_future_y = future_data_unscaled 
        
        # reset the $ for this run
        I_rem = 1.0
        total_cost_paid = 0.0
        
        # this is the mpc loop (88 steps)
        for t in range(H):
            
            # 1. PREPARE HISTORY
            # convert 1D list [p1, p2, ...] to 2D array [[p1], [p2], ...]
            history_2d_unscaled = np.array(current_history_unscaled).reshape(-1, 1)
            
            # Scale it, then reshape for the model: (1, seq_len, n_vars)
            history_scaled = self.scaler.transform(history_2d_unscaled)
            history_tensor = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 2. PREDICT (make a new plan)
            pred_y_scaled = self.forecast_model(history_tensor)
            pred_y_unscaled = self.scaler.inverse_transform(pred_y_scaled.cpu().numpy().reshape(1, H))

            # 3. PLAN
            # I need to calculate how many steps are left in this 88-step race
            remaining_steps = H - t 

            # Now, I build a NEW Gurobi model for *only* the remaining steps
            # This forces it to spend 100% of my money in the time i have left
            current_opt_model = AllocateModel(
                self.constraint.numpy(),
                self.configs.uncertainty_quantile,
                pred_len=remaining_steps # This is the shrinking number (88, 87, 86...)
            )

            # i still give it the full 88-step prediction (i have to, because of model we trained on SPO loss)
            current_opt_model.setObj(pred_y_unscaled.squeeze())

            # now when i solve, it will *only* make a plan for the remaining steps
            # and it will set all steps after that to 0
            sol, _ = current_opt_model.solve()
            
            # 4. ACT (but i only use the first step of the plan)
            a_star_t_plus_1 = sol[0] 
            
            if t == H - 1:
                amount_to_buy = I_rem # On the last step, buy everything left
            else:
                a_star_t_plus_1 = min(max(a_star_t_plus_1, 0.0), 1.0) # safety check
                amount_to_buy = a_star_t_plus_1 * I_rem
            
            # Get the *true* price for this step from our "answer key"
            true_price_this_step = true_future_y[t]
            cost_this_step = amount_to_buy * true_price_this_step
            total_cost_paid += cost_this_step
            I_rem = max(I_rem - amount_to_buy, 0.0) # Update budget
            
            # 5. UPDATE HISTORY (the feedback part)
            current_history_unscaled.pop(0) # Remove oldest price
            current_history_unscaled.append(true_future_y[t]) # Add new true price
        
        # --- end of the 88 steps ---
        
        # now grade this one simulation
        total_bought = 1.0 - I_rem # should be close to 1.0
        
        if total_bought < 1e-6:
             avg_cost_paid = 0 # something went wrong if we bought nothing
        else:
            avg_cost_paid = total_cost_paid / total_bought

        true_optimal_cost = np.min(true_future_y) # Best price in THIS 88-step window
        regret = avg_cost_paid - true_optimal_cost
        
        return regret

    def _save_results(self, metrics):
        # just save the json file
        res_path = os.path.join(self.exp_dir, 'mpc_avg_result.json')
        with open(res_path, 'w') as fout:
            metrics_serializable = {k: float(v) for k, v in metrics.items()}
            json.dump(metrics_serializable, fout, indent=4)
        print(f"Averaged MPC results saved to {res_path}")