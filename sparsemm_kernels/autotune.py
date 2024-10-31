import optuna

def autotune(bench_fn, args, param_ranges, n_trials=100):
    def objective(trial):
        params = {}
        for param_name, param_range in param_ranges.items():
            params[param_name] = trial.suggest_categorical(param_name, param_range)
        return bench_fn(*args, **params)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value