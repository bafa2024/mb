#!/usr/bin/env python3
"""
Development server runner for the NetCDF to Mapbox Converter
"""

import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print("✅ Loaded environment variables from .env")
else:
    print("⚠️  No .env file found. Using default settings.")
    print("   Copy .env.example to .env and add your Mapbox credentials.")

def main():
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = bool(int(os.getenv("DEBUG", 0)))
    
    # Check for required environment variables
    required_vars = ["MAPBOX_TOKEN", "MAPBOX_USERNAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your .env file.")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["temp_files", "static", "static/css", "static/js", "templates", "utils"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Create __init__.py for utils module
    init_file = Path("utils/__init__.py")
    if not init_file.exists():
        init_file.touch()
    
    print(f"""
🚀 Starting NetCDF to Mapbox Converter
    
   Server: http://{host}:{port}
   Debug:  {debug}
   User:   {os.getenv('MAPBOX_USERNAME')}
    
   Press Ctrl+C to stop
    """)
    
    # Run the server
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info"
    )

if __name__ == "__main__":
    main()