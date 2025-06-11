import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API credentials
api_key = os.getenv('KRAKEN_API_KEY')
api_secret = os.getenv('KRAKEN_API_SECRET')

print("\n🔑 Environment Variable Check:")
print("-" * 50)
print(f"API Key exists: {'✅' if api_key else '❌'}")
print(f"API Key length: {len(api_key) if api_key else 'N/A'}")
print(f"API Secret exists: {'✅' if api_secret else '❌'}")
print(f"API Secret length: {len(api_secret) if api_secret else 'N/A'}")

# Print the first and last 4 characters of each (if they exist)
if api_key:
    print(f"\nAPI Key: {api_key[:4]}...{api_key[-4:]}")
if api_secret:
    print(f"API Secret: {api_secret[:4]}...{api_secret[-4:]}") 