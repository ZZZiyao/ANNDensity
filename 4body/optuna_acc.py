"""
Optuna hyperparameter search for the cut+logit 6-feature acceptance ANN.

Searches: lambda2, lr, lr_final ratio, hidden architecture. Objective = best held-out
val NLL (lower=better; val_nll excludes the L2 term so different lambda2 are comparable).
Median-pruner kills bad trials early using intermediate val NLL.

PARALLEL workers: point every job at the SAME SQLite storage (ACC_OPTUNA_DB on the shared
RDS filesystem) with the SAME study name -> they cooperate on one study. Launch as a slurm
job array; each task runs ACC_OPTUNA_TRIALS trials.

Env: ACC_OPTUNA_DB (sqlite path), ACC_OPTUNA_STUDY, ACC_OPTUNA_TRIALS, ACC_SEARCH_EPOCHS
     (reduced epochs per trial for speed), ACC_NTRAIN (subsample for speed).
"""
import os
import optuna
import train_acc_6d as T

HERE = os.path.dirname(os.path.abspath(__file__))
DB      = os.environ.get("ACC_OPTUNA_DB", os.path.join(HERE, "optuna_acc.db"))
STUDY   = os.environ.get("ACC_OPTUNA_STUDY", "acc6d")
NTRIALS = int(os.environ.get("ACC_OPTUNA_TRIALS", 20))
EPOCHS  = int(os.environ.get("ACC_SEARCH_EPOCHS", 15000))   # proxy epochs per trial
NTRAIN  = int(os.environ.get("ACC_NTRAIN", 2000000))        # subsample for speed
NORM    = int(os.environ.get("ACC_NORM_SIZE", 1000000))

ARCHS = {
    "small":  [32, 16, 8],
    "paper":  [32, 64, 32, 8],
    "wide":   [64, 128, 64, 16],
    "deep":   [32, 64, 64, 32, 8],
}


def objective(trial):
    lam = trial.suggest_float("lambda2", 1e-4, 1.0, log=True)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    lr_ratio = trial.suggest_float("lr_final_ratio", 0.01, 1.0, log=True)
    arch = trial.suggest_categorical("arch", list(ARCHS.keys()))
    cfg = dict(lambda2=lam, lr=lr, lr_final=lr * lr_ratio, n_hidden=ARCHS[arch],
               max_epochs=EPOCHS, norm_size=NORM, ntrain=NTRAIN, check_every=200,
               seed=1, outfile=None)

    def prune_cb(epoch, val):
        trial.report(val, epoch)
        return trial.should_prune()

    res = T.train_model(cfg, verbose=False, prune_cb=prune_cb)
    return res["best_val"]


def main():
    storage = f"sqlite:///{DB}"
    study = optuna.create_study(
        study_name=STUDY, storage=storage, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=None),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3000))
    print(f"study '{STUDY}' on {DB}: {len(study.trials)} trials so far; running {NTRIALS} more")
    study.optimize(objective, n_trials=NTRIALS)
    print("\n=== best so far ===")
    print("value (val NLL):", study.best_value)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
