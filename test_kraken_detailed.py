#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import time
import hashlib
import hmac
import base64
import urllib.parse
import requests

# Load environment variables
load_dotenv()

# Get API credentials
API_KEY = os.getenv('KRAKEN_API_KEY')
API_SECRET = os.getenv('KRAKEN_API_SECRET')

def get_kraken_signature(urlpath, data, secret):
    """Generate Kraken API signature"""
    try:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    except Exception as e:
        print(f"Error generating signature: {e}")
        return None

def test_private_endpoint():
    """Test private API endpoint with detailed debugging"""
    
    print("\n🔍 Detailed Kraken API Test")
    print("-" * 50)
    
    # 1. Check credentials
    print("\n1. Checking API Credentials:")
    print(f"API Key exists: {'✅' if API_KEY else '❌'}")
    print(f"API Secret exists: {'✅' if API_SECRET else '❌'}")
    
    if not API_KEY or not API_SECRET:
        print("❌ Missing API credentials!")
        return
    
    # 2. Prepare request
    print("\n2. Preparing API Request:")
    endpoint = '/0/private/Balance'
    url = f"https://api.kraken.com{endpoint}"
    
    # Generate nonce
    nonce = str(int(time.time() * 1000))
    
    # Request data
    data = {
        "nonce": nonce
    }
    
    try:
        # 3. Generate signature
        print("\n3. Generating API Signature:")
        signature = get_kraken_signature(endpoint, data, API_SECRET)
        print(f"Signature generated: {'✅' if signature else '❌'}")
        
        if not signature:
            print("❌ Failed to generate signature!")
            return
        
        # 4. Prepare headers
        headers = {
            'API-Key': API_KEY,
            'API-Sign': signature
        }
        
        # 5. Make request
        print("\n4. Making API Request:")
        response = requests.post(url, headers=headers, data=data)
        
        # 6. Process response
        print("\n5. Processing Response:")
        print(f"Status Code: {response.status_code}")
        
        try:
            result = response.json()
            if 'error' in result and result['error']:
                print(f"❌ API Error: {result['error']}")
            else:
                print("✅ Success! Response received:")
                print(result)
        except Exception as e:
            print(f"❌ Failed to parse response: {e}")
            print(f"Raw response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_private_endpoint() 