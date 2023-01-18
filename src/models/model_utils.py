import json

def get_labels():
    with open("./deployment/app/static/assets/labels/labels.json") as f:
        labels = json.load(f)
    labels = {int(k): v for (k, v) in labels.items()}
    return labels
