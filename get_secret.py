from google.cloud import secretmanager

# Create the Secret Manager client.
gcp_secret_client = secretmanager.SecretManagerServiceClient()

# Build the complete resource name of the secret version.
secret_resource_name = f"projects/1063147579069/secrets/DATA_KEY_FILE/versions/1"

# Access the secret version.
response = gcp_secret_client.access_secret_version(name=secret_resource_name)

# Get the secret value.
secret_value = response.payload.data.decode("UTF-8")
