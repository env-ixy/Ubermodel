import random

NUM_MODELS = 20
K_FLOOR = 0.001

models = [[random.random(), random.random()] for _ in range(NUM_MODELS)]

def objective(model):
    x, y = model
    return -((x - 3)**2 + (y - 2)**2)

def assign_f(model):
    return model[0] + model[1]

def compute_K(models):
    """
    K = (1/n) * sum(|f_i - f_mean|)
    Mean absolute deviation, floored at K_FLOOR to prevent
    population freezing at small sizes.
    """
    f_values = [assign_f(m) for m in models]
    f_mean = sum(f_values) / len(f_values)
    k = sum(abs(f - f_mean) for f in f_values) / len(f_values)
    return max(k, K_FLOOR)

def make_groups(models, K):
    sorted_models = sorted(models, key=assign_f)
    groups = []
    current_group = [sorted_models[0]]
    group_start_f = assign_f(sorted_models[0])

    for m in sorted_models[1:]:
        if abs(assign_f(m) - group_start_f) < K:
            current_group.append(m)
        else:
            groups.append(current_group)
            current_group = [m]
            group_start_f = assign_f(m)

    if current_group:
        groups.append(current_group)

    return groups

with open("logs.txt", "a") as log:
    f_values = [assign_f(m) for m in models]
    K = compute_K(models)
    groups = make_groups(models, K)
    best_model = max(models, key=objective)
    best_score = objective(best_model)
    
    # Log models with their f values
    log.write("Models with f values:\n")
    for i, m in enumerate(models):
        log.write(f"  Model {i}: {m}, f = {assign_f(m):.4f}\n")
    
    # Log groups
    log.write(f"\nGroups (K = {K:.4f}):\n")
    for i, group in enumerate(groups):
        log.write(f"  Group {i}: {group}\n")
    
    # Log scores
    log.write(f"\nScores:\n")
    log.write(f"  Best Model: {best_model}, Score = {best_score:.4f}\n")
    log.write("End of cycle----------------------------\n")