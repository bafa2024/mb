#!/usr/bin/env python3
"""
Test script to verify Mapbox credentials and API access
"""

import requests
import sys

def test_mapbox_credentials(token, username):
    print("🔍 Testing Mapbox Credentials")
    print("=" * 50)
    print(f"Token: {token[:20]}...")
    print(f"Username: {username}")
    print()
    
    # Test 1: Validate token
    print("1. Testing token validity...")
    token_url = f"https://api.mapbox.com/tokens/v2?access_token={token}"
    resp = requests.get(token_url)
    
    if resp.status_code == 200:
        token_info = resp.json()
        print("   ✅ Token is valid!")
        
        # Check scopes
        scopes = token_info.get('token', {}).get('scopes', [])
        print(f"   Token scopes: {scopes}")
        
        required_scopes = ['tilesets:write', 'tilesets:read', 'tilesets:list']
        missing_scopes = [s for s in required_scopes if s not in scopes]
        
        if missing_scopes:
            print(f"   ❌ Missing required scopes: {missing_scopes}")
            print("   Please create a new token with these scopes at:")
            print("   https://account.mapbox.com/access-tokens/")
            return False
        else:
            print("   ✅ Token has all required scopes!")
    else:
        print(f"   ❌ Token validation failed: {resp.status_code}")
        print(f"   Response: {resp.text}")
        return False
    
    # Test 2: Check if username exists
    print("\n2. Testing username...")
    # Try to list tilesets (this will fail if username is wrong)
    list_url = f"https://api.mapbox.com/tilesets/v1/{username}?access_token={token}"
    resp = requests.get(list_url)
    
    if resp.status_code == 200:
        print("   ✅ Username is valid!")
    else:
        print(f"   ❌ Username test failed: {resp.status_code}")
        print(f"   Response: {resp.text}")
        if resp.status_code == 404:
            print("   The username might be incorrect. Check your username at:")
            print("   https://account.mapbox.com/")
        return False
    
    # Test 3: Test source upload credentials endpoint
    print("\n3. Testing source upload credentials...")
    source_id = "test_source_123"
    cred_url = f"https://api.mapbox.com/tilesets/v1/sources/{username}/{source_id}/upload-credentials?access_token={token}"
    resp = requests.post(cred_url)
    
    if resp.status_code == 200:
        print("   ✅ Can get upload credentials!")
        creds = resp.json()
        print(f"   S3 Bucket: {creds.get('bucket')}")
        print(f"   AWS Region: {creds.get('region', 'us-east-1')}")
    else:
        print(f"   ❌ Failed to get upload credentials: {resp.status_code}")
        print(f"   Response: {resp.text}")
        
        if resp.status_code == 401:
            print("\n   Possible issues:")
            print("   - Token doesn't have 'tilesets:write' scope")
            print("   - Token has expired")
            
        elif resp.status_code == 404:
            print("\n   Possible issues:")
            print("   - Username is incorrect")
            print("   - API endpoint has changed")
            
        return False
    
    # Test 4: List existing tilesets
    print("\n4. Listing your tilesets...")
    list_url = f"https://api.mapbox.com/tilesets/v1/{username}?access_token={token}&limit=5"
    resp = requests.get(list_url)
    
    if resp.status_code == 200:
        tilesets = resp.json()
        print(f"   ✅ Found {len(tilesets)} tilesets")
        for ts in tilesets[:3]:
            print(f"   - {ts.get('id')}: {ts.get('name')}")
    else:
        print(f"   ⚠️  Could not list tilesets: {resp.status_code}")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Your credentials are configured correctly.")
    print("\nYou can now upload NetCDF files to Mapbox!")
    return True


if __name__ == "__main__":
    print("Mapbox Credentials Tester")
    print("========================\n")
    
    # Get credentials
    token = input("Enter your Mapbox token (or press Enter to use .env): ").strip()
    if not token:
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            token = os.getenv("MAPBOX_TOKEN")
            if not token:
                print("❌ No token found in .env file")
                sys.exit(1)
        except ImportError:
            print("❌ python-dotenv not installed. Run: pip install python-dotenv")
            sys.exit(1)
    
    username = input("Enter your Mapbox username (or press Enter to use .env): ").strip()
    if not username:
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            username = os.getenv("MAPBOX_USERNAME")
            if not username:
                print("❌ No username found in .env file")
                sys.exit(1)
        except ImportError:
            pass
    
    print()
    success = test_mapbox_credentials(token, username)
    
    if not success:
        print("\n❌ Some tests failed. Please fix the issues above and try again.")
        sys.exit(1)