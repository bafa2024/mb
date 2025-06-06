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
import base64
import io
from PIL import Image

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

def create_recipe_for_netcdf(nc_path: str, tileset_id: str, username: str) -> Dict:
    """
    Create a Mapbox recipe for NetCDF data with RasterArray format.
    Supports both vector fields (u/v components) and scalar fields.
    """
    try:
        ds = xr.open_dataset(nc_path)
        
        # Identify variables and their types
        scalar_vars = []
        vector_pairs = []
        all_bands = {}
        band_index = 0
        
        # Common vector component patterns
        vector_patterns = [
            ('u', 'v'),  # u/v wind components
            ('u10', 'v10'),  # 10m wind components
            ('u_wind', 'v_wind'),
            ('uwind', 'vwind'),
            ('water_u', 'water_v'),  # ocean currents
            ('eastward_wind', 'northward_wind'),
        ]
        
        # Check for vector pairs
        var_names = list(ds.data_vars)
        processed_vars = set()
        
        for u_pattern, v_pattern in vector_patterns:
            u_matches = [v for v in var_names if u_pattern in v.lower() and v not in processed_vars]
            v_matches = [v for v in var_names if v_pattern in v.lower() and v not in processed_vars]
            
            if u_matches and v_matches:
                u_var = u_matches[0]
                v_var = v_matches[0]
                vector_pairs.append({
                    'u': u_var,
                    'v': v_var,
                    'name': 'wind' if 'wind' in u_var.lower() else 'flow'
                })
                processed_vars.add(u_var)
                processed_vars.add(v_var)
        
        # All other variables are scalar
        scalar_vars = [v for v in var_names if v not in processed_vars]
        
        # Build the RasterArray recipe
        recipe = {
            "version": 1,
            "layers": {
                "default": {
                    "source": f"mapbox://tileset-source/{username}/{tileset_id}",
                    "minzoom": 0,
                    "maxzoom": 14,
                    "raster_array": {
                        "bands": {}
                    }
                }
            }
        }
        
        # Add all variables as bands
        band_mapping = {}
        
        # First add vector components
        for vector_pair in vector_pairs:
            # U component
            band_index += 1
            band_name = f"{vector_pair['name']}_u"
            recipe["layers"]["default"]["raster_array"]["bands"][band_name] = {
                "band": band_index,
                "source_band": vector_pair['u']
            }
            band_mapping[vector_pair['u']] = band_name
            all_bands[band_name] = {
                "source": vector_pair['u'],
                "type": "vector_u",
                "stats": {
                    "min": float(ds[vector_pair['u']].min().values),
                    "max": float(ds[vector_pair['u']].max().values)
                }
            }
            
            # V component
            band_index += 1
            band_name = f"{vector_pair['name']}_v"
            recipe["layers"]["default"]["raster_array"]["bands"][band_name] = {
                "band": band_index,
                "source_band": vector_pair['v']
            }
            band_mapping[vector_pair['v']] = band_name
            all_bands[band_name] = {
                "source": vector_pair['v'],
                "type": "vector_v",
                "stats": {
                    "min": float(ds[vector_pair['v']].min().values),
                    "max": float(ds[vector_pair['v']].max().values)
                }
            }
        
        # Then add scalar variables
        for scalar_var in scalar_vars:
            band_index += 1
            # Clean variable name for band name
            band_name = re.sub(r'[^a-zA-Z0-9_]', '_', scalar_var.lower())
            
            recipe["layers"]["default"]["raster_array"]["bands"][band_name] = {
                "band": band_index,
                "source_band": scalar_var
            }
            band_mapping[scalar_var] = band_name
            all_bands[band_name] = {
                "source": scalar_var,
                "type": "scalar",
                "stats": {
                    "min": float(ds[scalar_var].min().values),
                    "max": float(ds[scalar_var].max().values)
                }
            }
        
        # Add tile configuration
        recipe["layers"]["default"]["tiles"] = {
            "buffer_size": 1,
            "tile_size": 512,
            "filter": ["all"],
            "resampling": "bilinear"
        }
        
        # Add metadata for easier reference
        recipe["metadata"] = {
            "band_mapping": band_mapping,
            "vector_pairs": vector_pairs,
            "scalar_variables": scalar_vars,
            "all_bands": all_bands,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": os.path.basename(nc_path)
        }
        
        ds.close()
        return recipe
        
    except Exception as e:
        print(f"Error creating recipe: {str(e)}")
        traceback.print_exc()
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

def prepare_netcdf_for_mapbox_advanced(ds: xr.Dataset, output_path: str):
    """
    Advanced preparation of NetCDF for Mapbox with proper coordinate handling
    """
    # Ensure proper coordinate names and CRS
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
    
    # Ensure all variables have proper CRS
    for var in ds.data_vars:
        if 'longitude' in ds[var].dims and 'latitude' in ds[var].dims:
            try:
                ds[var].rio.write_crs("EPSG:4326", inplace=True)
                ds[var].rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            except Exception as e:
                print(f"Warning: Could not set CRS for {var}: {e}")
    
    # Handle time dimension if present
    if 'time' in ds.dims and len(ds.time) > 1:
        # For now, select first time step
        # In production, you might want to create separate bands for each time
        ds = ds.isel(time=0)
    
    # Save with proper encoding
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            'dtype': 'float32',
            'scale_factor': 0.01,
            'add_offset': 0,
            '_FillValue': -9999
        }
    
    ds.to_netcdf(output_path, encoding=encoding)

def generate_preview_images(ds: xr.Dataset) -> Dict:
    """
    Generate base64-encoded preview images for visualization
    """
    preview_data = {}
    
    try:
        # Generate previews for first few variables
        for var in list(ds.data_vars)[:3]:
            da = ds[var]
            
            # Handle time dimension
            if 'time' in da.dims:
                da = da.isel(time=0)
            
            # Convert to numpy array
            data = da.values
            
            # Normalize to 0-255
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if data_max > data_min:
                normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(data, dtype=np.uint8)
            
            # Create image
            img = Image.fromarray(normalized, mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Get bounds
            if 'longitude' in da.dims and 'latitude' in da.dims:
                bounds = {
                    'west': float(da.longitude.min()),
                    'east': float(da.longitude.max()),
                    'south': float(da.latitude.min()),
                    'north': float(da.latitude.max())
                }
            else:
                bounds = {'west': -180, 'east': 180, 'south': -90, 'north': 90}
            
            preview_data[var] = {
                'data': img_base64,
                'bounds': bounds,
                'stats': {
                    'min': float(data_min),
                    'max': float(data_max),
                    'mean': float(np.nanmean(data))
                }
            }
            
    except Exception as e:
        print(f"Error generating previews: {e}")
    
    return preview_data

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
                    recipe = create_recipe_for_netcdf(str(prepared_path), tileset_id, username)
                    
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
                    response_data["visualization_url"] = f"/visualize/{tileset_id}"
                    
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

@app.post("/process-advanced")
async def process_file_advanced(
    file: UploadFile = File(...),
    mapbox_token: Optional[str] = Form(None),
    mapbox_username: Optional[str] = Form(None),
    upload_to_mapbox: bool = Form(True),
    create_recipe: bool = Form(True),
    enable_queries: bool = Form(False),
    temporal_start: Optional[str] = Form(None),
    temporal_end: Optional[str] = Form(None),
    spatial_bounds: Optional[str] = Form(None)  # Format: "west,south,east,north"
):
    """
    Advanced processing with query support and tileset management
    """
    filepath = None
    prepared_path = None
    
    try:
        # Use provided credentials or fall back to environment variables
        token = mapbox_token or MAPBOX_TOKEN
        username = mapbox_username or MAPBOX_USERNAME
        
        # Save uploaded NetCDF file
        filepath = TEMP_DIR / file.filename
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Open dataset for analysis
        ds = xr.open_dataset(str(filepath))
        
        # Apply queries if requested
        if enable_queries:
            # Temporal query
            if temporal_start or temporal_end:
                from tileset_management import TilesetQueryTools
                ds = TilesetQueryTools.query_temporal_range(ds, temporal_start, temporal_end)
            
            # Spatial query
            if spatial_bounds:
                bounds = [float(x) for x in spatial_bounds.split(',')]
                if len(bounds) == 4:
                    from tileset_management import TilesetQueryTools
                    ds = TilesetQueryTools.query_spatial_bounds(ds, *bounds)
        
        # Compute statistics
        from tileset_management import TilesetQueryTools
        statistics = TilesetQueryTools.compute_statistics(ds)
        
        # Prepare for Mapbox
        prepared_path = TEMP_DIR / f"prepared_{file.filename}"
        prepare_netcdf_for_mapbox_advanced(ds, str(prepared_path))
        
        # Generate preview images for variables
        preview_data = generate_preview_images(ds)
        
        response_data = {
            "success": True,
            "statistics": statistics,
            "preview_data": preview_data,
            "message": f"Successfully processed {file.filename}"
        }
        
        # Upload to Mapbox if requested
        if upload_to_mapbox and token and username:
            tileset_id = re.sub(r'[^a-zA-Z0-9_-]', '_', 
                              os.path.splitext(file.filename)[0].lower())[:32]
            
            # Create recipe
            recipe = create_recipe_for_netcdf(str(prepared_path), tileset_id, username)
            
            # Use tileset manager
            from tileset_management import MapboxTilesetManager
            manager = MapboxTilesetManager(token, username)
            
            result = manager.process_netcdf_to_tileset(str(prepared_path), tileset_id, recipe)
            
            if result['success']:
                response_data["mapbox_upload"] = True
                response_data["tileset_id"] = result['tileset_id']
                response_data["job_id"] = result.get('job_id')
                response_data["recipe"] = recipe
                response_data["visualization_url"] = f"/visualize/{tileset_id}"
                
                # Save recipe for later reference
                recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
                with open(recipe_path, 'w') as f:
                    json.dump(recipe, f, indent=2)
            else:
                response_data["mapbox_error"] = result.get('error')
        
        ds.close()
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error processing file: {str(e)}"
        }, status_code=500)

@app.get("/visualize/{tileset_id}")
async def visualize_tileset(request: Request, tileset_id: str):
    """
    Display multi-variable visualization page for a tileset
    """
    # Get recipe data if available
    recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
    if recipe_path.exists():
        with open(recipe_path, 'r') as f:
            recipe = json.load(f)
    else:
        recipe = {}
    
    # Use public token for visualization
    public_token = MAPBOX_PUBLIC_TOKEN or MAPBOX_TOKEN
    
    return templates.TemplateResponse("multi_variable_visualization.html", {
        "request": request,
        "tileset_id": f"{MAPBOX_USERNAME}.{tileset_id}",
        "mapbox_token": public_token,
        "recipe": recipe
    })

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

@app.get("/api/tileset-info/{tileset_id}")
async def get_tileset_info(tileset_id: str):
    """
    Get information about a tileset including available layers and statistics
    """
    try:
        # Load recipe if available
        recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
        if recipe_path.exists():
            with open(recipe_path, 'r') as f:
                recipe = json.load(f)
                
            return JSONResponse({
                "success": True,
                "recipe": recipe,
                "bands": recipe.get("metadata", {}).get("all_bands", {}),
                "vector_pairs": recipe.get("metadata", {}).get("vector_pairs", []),
                "scalar_variables": recipe.get("metadata", {}).get("scalar_variables", [])
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "Recipe not found"
            }, status_code=404)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)

@app.post("/api/query-tileset")
async def query_tileset(
    tileset_id: str = Form(...),
    query_type: str = Form(...),  # 'statistics', 'temporal', 'spatial'
    parameters: str = Form("{}")   # JSON string with query parameters
):
    """
    Perform queries on tileset data
    """
    try:
        params = json.loads(parameters)
        
        # This would typically query the actual tileset data
        # For demonstration, we'll return sample results
        
        if query_type == "statistics":
            return JSONResponse({
                "success": True,
                "query_type": query_type,
                "results": {
                    "temperature": {
                        "min": -10.5,
                        "max": 35.2,
                        "mean": 15.7,
                        "std": 8.3
                    },
                    "wind_speed": {
                        "min": 0,
                        "max": 25.5,
                        "mean": 8.2,
                        "std": 4.1
                    }
                }
            })
        
        elif query_type == "temporal":
            return JSONResponse({
                "success": True,
                "query_type": query_type,
                "results": {
                    "time_range": params.get("time_range", ["2024-01-01", "2024-12-31"]),
                    "data_points": 365,
                    "temporal_resolution": "daily"
                }
            })
        
        elif query_type == "spatial":
            bounds = params.get("bounds", [-180, -90, 180, 90])
            return JSONResponse({
                "success": True,
                "query_type": query_type,
                "results": {
                    "bounds": bounds,
                    "grid_points": 10000,
                    "spatial_resolution": "0.25 degrees"
                }
            })
        
        else:
            return JSONResponse({
                "success": False,
                "message": f"Unknown query type: {query_type}"
            }, status_code=400)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)

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