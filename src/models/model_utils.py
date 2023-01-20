import json

def get_labels():
    """
    Get labels associated with each plant disease class
    """
    with open("./deployment/app/static/assets/labels/labels.json") as f:
        labels = json.load(f)
    labels = {int(k): v for (k, v) in labels.items()}
    return labels


def check_prune_level(module):
    import torch

    sparsity_level = 100 * float(torch.sum(module.weight == 0) / module.weight.numel())
    print(f"Sparsity level of module {sparsity_level}")


def prune_model(ckpt_path):

    import torch
    from torch.nn.utils import prune
    from model import ImageClassification

    # Initialize model
    pruned_model = ImageClassification()
    pruned_model = pruned_model.load_from_checkpoint(ckpt_path)

    torch.save(pruned_model.state_dict(), "models/non_pruned_network.pt")

    import time

    tic = time.time()
    for _ in range(100):
        _ = pruned_model.model(torch.randn(100, 3, 28, 28))
    toc = time.time()
    print("Inference time (regular) x100:", toc - tic)

    # Finding parameters to prune
    parameters_to_prune = []
    for name, module in pruned_model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))
    parameters_to_prune = tuple(parameters_to_prune)

    # Pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3,
    )

    # making sparse
    [module.weight_mask.to_sparse() for (module, weight) in parameters_to_prune]
    # removing previous weights
    [prune.remove(*tup) for tup in parameters_to_prune]

    print("Checking prune level (Pruned)")
    for name, module in pruned_model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            check_prune_level(module)

    torch.save(pruned_model.state_dict(), "models/pruned_network.pt")

    pruned_model = ImageClassification()
    pruned_model.load_state_dict(torch.load("models/pruned_network.pt"))

    import time

    tic = time.time()
    for _ in range(100):
        _ = pruned_model.model(torch.randn(100, 3, 28, 28))
    toc = time.time()
    print("Inference time x100 (Pruned):", toc - tic)

    return pruned_model


if __name__ == "__main__":
    prune_model("models/exp1/LR0.00359-BS100/-epoch=77-val_acc=0.92.ckpt")
