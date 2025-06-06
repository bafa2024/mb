from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import xarray as xr
import rioxarray
import os
import requests
import boto3
import json
import time
import re
import numpy as np
from typing import Optional, Dict, List
import tempfile
import traceback
from pathlib import Path

app = FastAPI()

# Create necessary directories if they don't exist
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("temp_files").mkdir(exist_ok=True)  # Use local temp directory instead of /tmp

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load Mapbox credentials from environment variables
MAPBOX_TOKEN = os.getenv("MAPBOX_SECRET_TOKEN", os.getenv("MAPBOX_TOKEN"))
MAPBOX_PUBLIC_TOKEN = os.getenv("MAPBOX_PUBLIC_TOKEN")
MAPBOX_USERNAME = os.getenv("MAPBOX_USERNAME")

# Use a local temp directory that we have permissions for
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def create_recipe_for_netcdf(nc_path: str, tileset_id: str) -> Dict:
    """
    Create a Mapbox recipe for NetCDF data with vector and scalar fields.
    """
    try:
        ds = xr.open_dataset(nc_path)
        
        # Identify variables and their types
        scalar_vars = []
        vector_pairs = []
        
        # Common vector component patterns
        vector_patterns = [
            ('u', 'v'),  # u/v wind components
            ('u10', 'v10'),  # 10m wind components
            ('u_wind', 'v_wind'),
            ('uwind', 'vwind'),
            ('water_u', 'water_v'),  # ocean currents
        ]
        
        # Check for vector pairs
        var_names = list(ds.data_vars)
        for u_pattern, v_pattern in vector_patterns:
            u_vars = [v for v in var_names if u_pattern in v.lower()]
            v_vars = [v for v in var_names if v_pattern in v.lower()]
            
            if u_vars and v_vars:
                vector_pairs.append({
                    'u': u_vars[0],
                    'v': v_vars[0],
                    'name': 'wind' if 'wind' in u_vars[0].lower() else 'flow'
                })
        
        # All other variables are scalar
        vector_components = []
        for pair in vector_pairs:
            vector_components.extend([pair['u'], pair['v']])
        
        scalar_vars = [v for v in var_names if v not in vector_components]
        
        # Build the recipe
        recipe = {
            "version": 1,
            "layers": {}
        }
        
        # Add vector layers
        for i, vector_pair in enumerate(vector_pairs):
            layer_name = f"{vector_pair['name']}_{i}"
            recipe["layers"][layer_name] = {
                "source": f"mapbox://tileset-source/{MAPBOX_USERNAME}/{tileset_id}",
                "minzoom": 0,
                "maxzoom": 12,
                "features": {
                    "attributes": {
                        "allowed_output": [
                            vector_pair['u'],
                            vector_pair['v'],
                            "speed",
                            "direction"
                        ],
                        "set": {
                            "speed": [
                                "case",
                                ["all", 
                                    ["has", vector_pair['u']], 
                                    ["has", vector_pair['v']]
                                ],
                                ["sqrt", 
                                    ["+", 
                                        ["^", ["get", vector_pair['u']], 2],
                                        ["^", ["get", vector_pair['v']], 2]
                                    ]
                                ],
                                None
                            ],
                            "direction": [
                                "case",
                                ["all", 
                                    ["has", vector_pair['u']], 
                                    ["has", vector_pair['v']]
                                ],
                                ["*", 180, ["/", ["atan2", ["get", vector_pair['v']], ["get", vector_pair['u']]], 3.14159]],
                                None
                            ]
                        }
                    }
                }
            }
        
        # Add scalar layers
        for scalar_var in scalar_vars:
            # Clean variable name for layer name
            layer_name = re.sub(r'[^a-zA-Z0-9_]', '_', scalar_var.lower())
            
            recipe["layers"][layer_name] = {
                "source": f"mapbox://tileset-source/{MAPBOX_USERNAME}/{tileset_id}",
                "minzoom": 0,
                "maxzoom": 12,
                "features": {
                    "attributes": {
                        "allowed_output": [scalar_var]
                    }
                }
            }
        
        # Add tile configuration
        recipe["tiles"] = {
            "raster_array": {
                "resampling": "bilinear",
                "overviews": [2, 4, 8, 16, 32],
                "pixel_type": "float32",
                "bands": {}
            }
        }
        
        # Configure bands for all variables
        all_vars = scalar_vars + vector_components
        for i, var in enumerate(all_vars):
            recipe["tiles"]["raster_array"]["bands"][var] = {
                "band_index": i + 1,
                "source_band": var,
                "statistics": {
                    "minimum": float(ds[var].min().values) if var in ds else -9999,
                    "maximum": float(ds[var].max().values) if var in ds else 9999
                }
            }
        
        ds.close()  # Close the dataset
        return recipe
        
    except Exception as e:
        print(f"Error creating recipe: {str(e)}")
        raise

def prepare_netcdf_for_mapbox(nc_path: str, output_path: str) -> Dict:
    """
    Prepare NetCDF file for Mapbox upload with proper formatting.
    Returns metadata about the prepared file.
    """
    try:
        ds = xr.open_dataset(nc_path)
        
        # Ensure proper coordinate names
        coord_mapping = {
            'longitude': ['lon', 'long', 'x', 'LON', 'LONGITUDE'],
            'latitude': ['lat', 'y', 'LAT', 'LATITUDE'],
            'time': ['time', 'TIME', 't', 'datetime', 'DATE']
        }
        
        for standard_name, alternatives in coord_mapping.items():
            for alt in alternatives:
                if alt in ds.dims or alt in ds.coords:
                    if alt != standard_name:
                        ds = ds.rename({alt: standard_name})
                    break
        
        # Ensure CRS is set for data variables
        for var in ds.data_vars:
            if 'longitude' in ds[var].dims and 'latitude' in ds[var].dims:
                try:
                    ds[var].rio.write_crs("EPSG:4326", inplace=True)
                    ds[var].rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
                except Exception as e:
                    print(f"Warning: Could not set CRS for {var}: {e}")
        
        # Save the prepared NetCDF
        ds.to_netcdf(output_path)
        
        # Extract metadata
        metadata = {
            "dimensions": dict(ds.dims),
            "variables": list(ds.data_vars),
            "scalar_vars": [],
            "vector_pairs": []
        }
        
        # Identify variable types
        var_names = list(ds.data_vars)
        vector_patterns = [('u', 'v'), ('u10', 'v10'), ('u_', 'v_')]
        
        for u_pat, v_pat in vector_patterns:
            u_matches = [v for v in var_names if u_pat in v.lower()]
            v_matches = [v for v in var_names if v_pat in v.lower()]
            if u_matches and v_matches:
                metadata["vector_pairs"].append({
                    "u": u_matches[0],
                    "v": v_matches[0],
                    "type": "wind" if "wind" in u_matches[0].lower() else "vector"
                })
        
        vector_components = []
        for pair in metadata["vector_pairs"]:
            vector_components.extend([pair["u"], pair["v"]])
        
        metadata["scalar_vars"] = [v for v in var_names if v not in vector_components]
        
        ds.close()  # Close the dataset
        return metadata
        
    except Exception as e:
        print(f"Error preparing NetCDF: {str(e)}")
        raise

def upload_to_mapbox_mts(token: str, username: str, nc_path: str, tileset_id: str, recipe: Dict) -> Dict:
    """
    Upload NetCDF to Mapbox using MTS API with recipe.
    """
    try:
        # Step 1: Create tileset source
        source_id = f"{tileset_id}_source"
        
        # Get S3 credentials for source upload
        cred_url = f"https://api.mapbox.com/tilesets/v1/sources/{username}/{source_id}/upload-credentials?access_token={token}"
        cred_resp = requests.post(cred_url)
        
        if cred_resp.status_code != 200:
            raise Exception(f"Failed to get upload credentials: {cred_resp.text}")
        
        creds = cred_resp.json()
        
        # Upload to S3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=creds['accessKeyId'],
            aws_secret_access_key=creds['secretAccessKey'],
            aws_session_token=creds['sessionToken'],
            region_name='us-east-1'
        )
        
        with open(nc_path, 'rb') as f:
            s3_client.put_object(
                Bucket=creds['bucket'],
                Key=creds['key'],
                Body=f
            )
        
        # Step 2: Create tileset with recipe
        tileset_url = f"https://api.mapbox.com/tilesets/v1/{username}.{tileset_id}?access_token={token}"
        tileset_data = {
            "recipe": recipe,
            "name": f"Weather Data - {tileset_id}",
            "description": "Multi-variable weather data with vector and scalar fields"
        }
        
        create_resp = requests.post(tileset_url, json=tileset_data)
        if create_resp.status_code not in [200, 201]:
            # Try updating if already exists
            update_resp = requests.patch(tileset_url, json={"recipe": recipe})
            if update_resp.status_code != 200:
                raise Exception(f"Failed to create/update tileset: {update_resp.text}")
        
        # Step 3: Publish tileset
        publish_url = f"https://api.mapbox.com/tilesets/v1/{username}.{tileset_id}/publish?access_token={token}"
        publish_resp = requests.post(publish_url)
        
        if publish_resp.status_code != 200:
            raise Exception(f"Failed to publish tileset: {publish_resp.text}")
        
        job_id = publish_resp.json().get('jobId')
        
        return {
            "tileset_id": f"{username}.{tileset_id}",
            "source_id": f"{username}/{source_id}",
            "job_id": job_id,
            "recipe": recipe
        }
        
    except Exception as e:
        print(f"Error uploading to Mapbox: {str(e)}")
        raise

@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    mapbox_token: Optional[str] = Form(None),
    mapbox_username: Optional[str] = Form(None),
    upload_to_mapbox: bool = Form(False),
    create_recipe: bool = Form(False)
):
    filepath = None
    prepared_path = None
    
    try:
        # Use provided credentials or fall back to environment variables
        token = mapbox_token or MAPBOX_TOKEN
        username = mapbox_username or MAPBOX_USERNAME
        
        # Save uploaded NetCDF file to our temp directory
        print(f"Processing file: {file.filename}")
        
        # Use our local temp directory
        filepath = TEMP_DIR / file.filename
        
        # Save the file
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
            print(f"Saved file to: {filepath}, size: {len(content)} bytes")
        
        # Prepare NetCDF
        prepared_path = TEMP_DIR / f"prepared_{file.filename}"
        metadata = prepare_netcdf_for_mapbox(str(filepath), str(prepared_path))
        
        # Create response
        response_data = {
            "success": True,
            "metadata": metadata,
            "message": f"Successfully processed {file.filename}"
        }
        
        # Generate GeoTIFF for each variable (for preview/download)
        ds = xr.open_dataset(str(prepared_path))
        generated_files = []
        
        for var in metadata["variables"][:3]:  # Limit to first 3 for demo
            if var in ds.data_vars:
                da = ds[var]
                if 'time' in da.dims:
                    da = da.isel(time=0)
                
                out_tif = TEMP_DIR / f"{var}.tif"
                try:
                    da.rio.to_raster(str(out_tif))
                    generated_files.append(var)
                except Exception as e:
                    print(f"Warning: Could not create GeoTIFF for {var}: {e}")
        
        ds.close()  # Close the dataset
        
        response_data["generated_files"] = generated_files
        
        # Upload to Mapbox if requested
        if upload_to_mapbox and create_recipe:
            if not token or not username:
                response_data["message"] += "<br>⚠️ Mapbox upload requires credentials."
            else:
                try:
                    # Create tileset ID
                    tileset_id = re.sub(r'[^a-zA-Z0-9_-]', '_', 
                                      os.path.splitext(file.filename)[0].lower())[:32]
                    
                    # Create recipe
                    recipe = create_recipe_for_netcdf(str(prepared_path), tileset_id)
                    
                    # Upload with MTS
                    upload_result = upload_to_mapbox_mts(token, username, str(prepared_path), tileset_id, recipe)
                    
                    response_data["mapbox_upload"] = True
                    response_data["tileset_id"] = upload_result["tileset_id"]
                    response_data["job_id"] = upload_result["job_id"]
                    response_data["recipe"] = recipe
                    response_data["message"] += f"""
                    <br><br>✅ Successfully uploaded to Mapbox!
                    <br>Tileset ID: <code>{upload_result['tileset_id']}</code>
                    <br>Job ID: <code>{upload_result['job_id']}</code>
                    <br><br>The tileset includes:
                    <br>• {len(metadata['scalar_vars'])} scalar fields: {', '.join(metadata['scalar_vars'][:3])}...
                    <br>• {len(metadata['vector_pairs'])} vector fields: {', '.join([p['type'] for p in metadata['vector_pairs']])}
                    """
                    
                    # Save recipe for download
                    recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
                    with open(recipe_path, 'w') as f:
                        json.dump(recipe, f, indent=2)
                    
                    response_data["recipe_download"] = f"/download-recipe/{tileset_id}"
                    
                except Exception as e:
                    response_data["message"] += f"<br>⚠️ Mapbox upload failed: {str(e)}"
                    response_data["mapbox_error"] = str(e)
        
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error processing file: {str(e)}"
        }, status_code=500)
    
    finally:
        # Clean up temporary files after a delay (optional)
        # You might want to keep them for download purposes
        pass

@app.get("/download-tif/{variable}")
async def download_tif(variable: str):
    """Download a specific variable as GeoTIFF."""
    tif_path = TEMP_DIR / f"{variable}.tif"
    if tif_path.exists():
        return FileResponse(
            str(tif_path),
            filename=f"{variable}.tif",
            media_type="image/tiff"
        )
    return JSONResponse({"message": "GeoTIFF not found."}, status_code=404)

@app.get("/download-recipe/{tileset_id}")
async def download_recipe(tileset_id: str):
    """Download the Mapbox recipe JSON."""
    recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
    if recipe_path.exists():
        return FileResponse(
            str(recipe_path),
            filename=f"recipe_{tileset_id}.json",
            media_type="application/json"
        )
    return JSONResponse({"message": "Recipe not found."}, status_code=404)

@app.get("/check-job/{job_id}")
async def check_job(job_id: str, username: str = None, token: str = None):
    """Check the status of a Mapbox tileset publish job."""
    try:
        username = username or MAPBOX_USERNAME
        token = token or MAPBOX_TOKEN
        
        status_url = f"https://api.mapbox.com/tilesets/v1/{username}/jobs/{job_id}?access_token={token}"
        resp = requests.get(status_url)
        
        if resp.status_code == 200:
            return JSONResponse(resp.json())
        else:
            return JSONResponse({"error": "Failed to check job status"}, status_code=resp.status_code)
            
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/demo")
async def demo_page(request: Request):
    """Demo page showing how to visualize the tileset in Mapbox GL JS."""
    return templates.TemplateResponse("demo.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mapbox_configured": bool(MAPBOX_TOKEN and MAPBOX_USERNAME),
        "temp_dir": str(TEMP_DIR),
        "temp_dir_exists": TEMP_DIR.exists(),
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK)
    }

# Cleanup old temp files on startup (optional)
def cleanup_old_files():
    """Remove temp files older than 1 hour."""
    try:
        import time
        current_time = time.time()
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > 3600:  # 1 hour
                    file_path.unlink()
                    print(f"Cleaned up old file: {file_path}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Run cleanup on startup
cleanup_old_files()