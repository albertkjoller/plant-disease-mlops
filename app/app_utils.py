from google.cloud import storage
import ndjson

def get_labels():
    from google.cloud import storage
    import ndjson
    client = storage.Client('plant-disease-mlops')
    bucket = client.get_bucket('plant-disease-labels_bucket')
    blob = bucket.blob('labels.json')
    labels = ndjson.loads(blob.download_as_string())[0]
    labels = {int(k):v for k,v in labels.items()}
    return labels
