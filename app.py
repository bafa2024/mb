from fastapi import FastAPI, UploadFile, File, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import xarray as xr
import rioxarray
import os
import requests
import boto3
import json
import time
import re
import numpy as np
from typing import Optional, Dict, List, Tuple
import tempfile
import traceback
from pathlib import Path
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import asyncio
from datetime import datetime
import rasterio
from rasterio.transform import from_bounds
from rasterio.plot import show

# Import custom modules
from utils.recipe_generator import create_enhanced_recipe_for_netcdf
from utils.query_tools import EnhancedTilesetQueryTools
from tileset_management import MapboxTilesetManager

app = FastAPI(title="NetCDF to Mapbox Converter", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("temp_files").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load Mapbox credentials
MAPBOX_TOKEN = os.getenv("MAPBOX_SECRET_TOKEN", os.getenv("MAPBOX_TOKEN"))
MAPBOX_PUBLIC_TOKEN = os.getenv("MAPBOX_PUBLIC_TOKEN")
MAPBOX_USERNAME = os.getenv("MAPBOX_USERNAME")

# Local temp directory
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main upload interface"""
    return templates.TemplateResponse("index.html", {"request": request})

def safe_compute_stats(data_array):
    """Safely compute statistics handling dtype issues"""
    try:
        # Convert to float to avoid dtype issues
        data = data_array.values.astype(np.float32)
        
        # Remove NaN values for statistics
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) > 0:
            return {
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)) if len(valid_data) > 1 else 0.0
            }
        else:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0
            }
    except Exception as e:
        print(f"Error computing statistics: {e}")
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0
        }

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
        
        # Fix coordinate dtypes if needed
        if 'longitude' in ds.coords:
            ds['longitude'] = ds['longitude'].astype('float64')
        if 'latitude' in ds.coords:
            ds['latitude'] = ds['latitude'].astype('float64')
        
        # Ensure CRS is set for data variables
        for var in ds.data_vars:
            if 'longitude' in ds[var].dims and 'latitude' in ds[var].dims:
                try:
                    # Convert data to float32 to avoid dtype issues
                    ds[var] = ds[var].astype('float32')
                    
                    # Set CRS
                    ds[var].rio.write_crs("EPSG:4326", inplace=True)
                    ds[var].rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
                except Exception as e:
                    print(f"Warning: Could not set CRS for {var}: {e}")
        
        # Handle time dimension if present
        if 'time' in ds.dims and len(ds.time) > 1:
            # For now, select first time step
            # You can modify this to create separate bands for each time
            ds = ds.isel(time=0)
        
        # Handle NaN values - replace with a fill value
        for var in ds.data_vars:
            data_array = ds[var]
            if np.isnan(data_array).any():
                print(f"Warning: {var} contains NaN values, replacing with -9999")
                ds[var] = data_array.fillna(-9999)
        
        # Save with proper encoding
        encoding = {}
        for var in ds.data_vars:
            # Ensure float32 dtype
            if ds[var].dtype != np.float32:
                ds[var] = ds[var].astype(np.float32)
            
            encoding[var] = {
                'dtype': 'float32',
                '_FillValue': -9999,
                'zlib': True,
                'complevel': 4
            }
        
        # Save the prepared NetCDF
        ds.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
        
        # Extract metadata
        metadata = {
            "dimensions": dict(ds.dims),
            "variables": list(ds.data_vars),
            "scalar_vars": [],
            "vector_pairs": []
        }
        
        # Identify variable types
        var_names = list(ds.data_vars)
        vector_patterns = [
            ('u', 'v'), ('u10', 'v10'), ('u_', 'v_'), 
            ('water_u', 'water_v'), ('eastward', 'northward')
        ]
        
        processed_vars = set()
        for u_pat, v_pat in vector_patterns:
            u_matches = [v for v in var_names if u_pat in v.lower() and v not in processed_vars]
            v_matches = [v for v in var_names if v_pat in v.lower() and v not in processed_vars]
            if u_matches and v_matches:
                # Try to match pairs
                for u_var in u_matches:
                    for v_var in v_matches:
                        # Check if they form a pair
                        if u_var.replace(u_pat, '') == v_var.replace(v_pat, ''):
                            metadata["vector_pairs"].append({
                                "u": u_var,
                                "v": v_var,
                                "type": "wind" if "wind" in u_var.lower() else "vector"
                            })
                            processed_vars.add(u_var)
                            processed_vars.add(v_var)
                            break
        
        metadata["scalar_vars"] = [v for v in var_names if v not in processed_vars]
        
        ds.close()
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

def create_preview_from_variable(ds: xr.Dataset, variable: str, colormap: str = 'viridis') -> str:
    """Create a base64 PNG preview from a variable with proper dtype handling"""
    try:
        da = ds[variable]
        
        # Handle time dimension
        if 'time' in da.dims:
            da = da.isel(time=0)
        
        # Get data and ensure it's float
        data = da.values.astype(np.float32)
        
        # Handle NaN values
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize data for better visualization
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [2, 98])
        else:
            vmin, vmax = 0, 1
        
        # Create the plot
        im = ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(variable, rotation=270, labelpad=20)
        
        # Add title
        ax.set_title(f'{variable} Visualization', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
        
    except Exception as e:
        print(f"Error creating preview for {variable}: {e}")
        return None

@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    mapbox_token: Optional[str] = Form(None),
    mapbox_username: Optional[str] = Form(None),
    upload_to_mapbox: bool = Form(True),
    create_recipe: bool = Form(True)
):
    """Process NetCDF file and optionally upload to Mapbox"""
    filepath = None
    prepared_path = None
    
    try:
        # Use provided credentials or fall back to environment variables
        token = mapbox_token or MAPBOX_TOKEN
        username = mapbox_username or MAPBOX_USERNAME
        
        # Save uploaded file
        filepath = TEMP_DIR / file.filename
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Prepare NetCDF
        prepared_path = TEMP_DIR / f"prepared_{file.filename}"
        metadata = prepare_netcdf_for_mapbox(str(filepath), str(prepared_path))
        
        # Create response
        response_data = {
            "success": True,
            "metadata": metadata,
            "message": f"Successfully processed {file.filename}"
        }
        
        # Generate previews
        ds = xr.open_dataset(str(prepared_path))
        preview_data = {}
        
        for var in metadata["variables"][:6]:  # Limit to first 6 variables
            if var in ds.data_vars:
                preview_base64 = create_preview_from_variable(ds, var)
                if preview_base64:
                    stats = safe_compute_stats(ds[var])
                    preview_data[var] = {
                        'preview': preview_base64,
                        'stats': stats
                    }
        
        ds.close()
        response_data["previews"] = preview_data
        
        # Generate GeoTIFF files for download/preview
        ds = xr.open_dataset(str(prepared_path))
        generated_files = []
        
        for var in metadata["variables"][:6]:  # Limit to first 6 variables
            if var in ds.data_vars:
                da = ds[var]
                if 'time' in da.dims:
                    da = da.isel(time=0)
                
                out_tif = TEMP_DIR / f"{var}.tif"
                try:
                    # Ensure data is float32
                    da = da.astype('float32')
                    # Fill NaN values
                    da = da.fillna(-9999)
                    # Save as GeoTIFF
                    da.rio.to_raster(str(out_tif))
                    generated_files.append(var)
                except Exception as e:
                    print(f"Warning: Could not create GeoTIFF for {var}: {e}")
        
        ds.close()
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
                    
                    # Create enhanced recipe
                    recipe = create_enhanced_recipe_for_netcdf(str(prepared_path), tileset_id, username)
                    
                    # Upload with MTS
                    upload_result = upload_to_mapbox_mts(token, username, str(prepared_path), tileset_id, recipe)
                    
                    response_data["mapbox_upload"] = True
                    response_data["tileset_id"] = upload_result["tileset_id"]
                    response_data["job_id"] = upload_result["job_id"]
                    response_data["recipe"] = recipe
                    response_data["visualization_url"] = f"/visualize-advanced/{tileset_id}"
                    
                    # Save recipe for later reference
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

@app.get("/visualize-advanced/{tileset_id}")
async def visualize_advanced(request: Request, tileset_id: str):
    """Advanced visualization page with full multi-variable support"""
    recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
    if recipe_path.exists():
        with open(recipe_path, 'r') as f:
            recipe = json.load(f)
    else:
        recipe = {}
    
    # Extract visualization configuration from recipe
    viz_config = {
        "tileset_id": f"{MAPBOX_USERNAME}.{tileset_id}",
        "mapbox_token": MAPBOX_PUBLIC_TOKEN or MAPBOX_TOKEN,
        "recipe": recipe,
        "variables": [],
        "vector_fields": []
    }
    
    # Parse available variables
    bands_info = recipe.get('metadata', {}).get('bands_info', {})
    for band_name, info in bands_info.items():
        if info['type'] == 'scalar':
            viz_config['variables'].append({
                'name': band_name,
                'display_name': info.get('long_name', band_name),
                'units': info.get('units', ''),
                'range': [info['stats']['min'], info['stats']['max']],
                'band_index': info['band_index']
            })
    
    # Parse vector fields
    vector_pairs = recipe.get('metadata', {}).get('vector_pairs', [])
    for pair in vector_pairs:
        u_band = f"{pair['name']}_u"
        v_band = f"{pair['name']}_v"
        if u_band in bands_info and v_band in bands_info:
            viz_config['vector_fields'].append({
                'name': pair['name'],
                'u_band': u_band,
                'v_band': v_band,
                'u_index': bands_info[u_band]['band_index'],
                'v_index': bands_info[v_band]['band_index'],
                'units': bands_info[u_band].get('units', 'm/s')
            })
    
    return templates.TemplateResponse("advanced_visualization.html", {
        "request": request,
        "config": viz_config
    })

@app.get("/api/wind-grid/{tileset_id}")
async def get_wind_grid(tileset_id: str, 
                       west: float = -180, 
                       south: float = -90, 
                       east: float = 180, 
                       north: float = 90,
                       resolution: int = 50):
    """Get wind grid data for particle visualization"""
    try:
        # Load the recipe to get variable mappings
        recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
        if not recipe_path.exists():
            return JSONResponse({
                "error": "Recipe not found for tileset"
            }, status_code=404)
        
        with open(recipe_path, 'r') as f:
            recipe = json.load(f)
        
        # Find wind variables
        vector_pairs = recipe.get('metadata', {}).get('vector_pairs', [])
        if not vector_pairs:
            return JSONResponse({
                "error": "No wind data found in tileset"
            }, status_code=404)
        
        # For demo, return simulated wind grid
        # In production, query actual tileset data
        wind_pair = vector_pairs[0]
        
        # Create grid
        lats = np.linspace(south, north, resolution)
        lons = np.linspace(west, east, resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Simulate wind field
        u_grid = 10 * np.sin(lon_grid * 0.05) * np.cos(lat_grid * 0.05)
        v_grid = 10 * np.cos(lon_grid * 0.05) * np.sin(lat_grid * 0.05)
        
        return JSONResponse({
            "success": True,
            "grid": {
                "lats": lats.tolist(),
                "lons": lons.tolist(),
                "shape": [resolution, resolution]
            },
            "u_component": u_grid.tolist(),
            "v_component": v_grid.tolist(),
            "metadata": {
                "u_variable": wind_pair['u'],
                "v_variable": wind_pair['v'],
                "units": "m/s",
                "bounds": {
                    "west": west,
                    "east": east,
                    "south": south,
                    "north": north
                }
            }
        })
        
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

@app.post("/api/query-point")
async def query_point_data(
    tileset_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    variables: str = Form(None)
):
    """Query data values at a specific point"""
    try:
        # In production, query actual tileset data at the point
        # For now, return simulated values
        
        recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
        if recipe_path.exists():
            with open(recipe_path, 'r') as f:
                recipe = json.load(f)
        else:
            recipe = {}
        
        requested_vars = variables.split(',') if variables else []
        
        results = {
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "values": {}
        }
        
        # Simulate values based on location
        for var in requested_vars:
            if 'temp' in var.lower():
                # Temperature varies with latitude
                value = 20 + 15 * np.cos(np.radians(latitude))
                results["values"][var] = {
                    "value": round(value, 2),
                    "units": "°C"
                }
            elif var.endswith('_u') or 'u10' in var:
                # U wind component
                value = 10 * np.sin(np.radians(longitude))
                results["values"][var] = {
                    "value": round(value, 2),
                    "units": "m/s"
                }
            elif var.endswith('_v') or 'v10' in var:
                # V wind component
                value = 10 * np.cos(np.radians(longitude))
                results["values"][var] = {
                    "value": round(value, 2),
                    "units": "m/s"
                }
            else:
                # Generic scalar
                value = np.random.randn() * 10 + 50
                results["values"][var] = {
                    "value": round(value, 2),
                    "units": "unknown"
                }
        
        # Add wind speed and direction if both components available
        if any('_u' in v for v in requested_vars) and any('_v' in v for v in requested_vars):
            u_val = next((v["value"] for k, v in results["values"].items() if '_u' in k), 0)
            v_val = next((v["value"] for k, v in results["values"].items() if '_v' in k), 0)
            
            speed = np.sqrt(u_val**2 + v_val**2)
            direction = np.arctan2(v_val, u_val) * 180 / np.pi
            
            results["values"]["wind_speed"] = {
                "value": round(speed, 2),
                "units": "m/s"
            }
            results["values"]["wind_direction"] = {
                "value": round(direction, 1),
                "units": "degrees"
            }
        
        return JSONResponse(results)
        
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

@app.post("/api/export-visualization")
async def export_visualization(
    tileset_id: str = Form(...),
    variables: str = Form(...),
    bounds: str = Form(...),
    format: str = Form("geotiff"),
    resolution: int = Form(100)
):
    """Export visualization data in various formats"""
    try:
        # Parse bounds
        bounds_list = [float(x) for x in bounds.split(',')]
        if len(bounds_list) != 4:
            raise ValueError("Invalid bounds format")
        
        west, south, east, north = bounds_list
        requested_vars = variables.split(',')
        
        # Create synthetic data for demo
        # In production, extract from actual tileset
        lats = np.linspace(south, north, resolution)
        lons = np.linspace(west, east, resolution)
        
        export_data = {}
        
        for var in requested_vars:
            if 'temp' in var.lower():
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                data = 20 + 15 * np.cos(np.radians(lat_grid))
                export_data[var] = data
            elif var.endswith('_u'):
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                data = 10 * np.sin(np.radians(lon_grid) * 0.1)
                export_data[var] = data
            elif var.endswith('_v'):
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                data = 10 * np.cos(np.radians(lat_grid) * 0.1)
                export_data[var] = data
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "geotiff":
            output_file = TEMP_DIR / f"export_{tileset_id}_{timestamp}.tif"
            
            transform = from_bounds(west, south, east, north, resolution, resolution)
            
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=resolution,
                width=resolution,
                count=len(export_data),
                dtype='float32',
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                for i, (var_name, data) in enumerate(export_data.items(), 1):
                    dst.write(data.astype('float32'), i)
                    dst.set_band_description(i, var_name)
            
            return FileResponse(
                str(output_file),
                filename=f"weather_data_{timestamp}.tif",
                media_type="image/tiff"
            )
            
        elif format == "netcdf":
            output_file = TEMP_DIR / f"export_{tileset_id}_{timestamp}.nc"
            
            ds = xr.Dataset()
            for var_name, data in export_data.items():
                ds[var_name] = xr.DataArray(
                    data,
                    dims=['latitude', 'longitude'],
                    coords={'latitude': lats, 'longitude': lons}
                )
            
            ds.attrs['title'] = f'Exported data from tileset {tileset_id}'
            ds.attrs['created'] = datetime.now().isoformat()
            ds.attrs['bounds'] = bounds
            
            ds.to_netcdf(output_file)
            
            return FileResponse(
                str(output_file),
                filename=f"weather_data_{timestamp}.nc",
                media_type="application/x-netcdf"
            )
            
        elif format == "json":
            json_data = {
                "metadata": {
                    "tileset_id": tileset_id,
                    "bounds": {
                        "west": west,
                        "east": east,
                        "south": south,
                        "north": north
                    },
                    "resolution": resolution,
                    "variables": requested_vars,
                    "created": datetime.now().isoformat()
                },
                "coordinates": {
                    "latitudes": lats.tolist(),
                    "longitudes": lons.tolist()
                },
                "data": {}
            }
            
            for var_name, data in export_data.items():
                json_data["data"][var_name] = data.tolist()
            
            return JSONResponse(json_data)
            
        else:
            return JSONResponse({
                "error": f"Unsupported format: {format}"
            }, status_code=400)
            
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

@app.websocket("/ws/realtime-data/{tileset_id}")
async def websocket_realtime_data(websocket: WebSocket, tileset_id: str):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                bounds = data.get("bounds")
                variables = data.get("variables", [])
                
                # Stream simulated data updates
                while True:
                    update_data = {
                        "type": "data_update",
                        "timestamp": datetime.now().isoformat(),
                        "data": {}
                    }
                    
                    for var in variables:
                        if 'wind' in var:
                            update_data["data"][var] = {
                                "value": np.random.randn() * 5 + 10,
                                "direction": np.random.rand() * 360
                            }
                        else:
                            update_data["data"][var] = {
                                "value": np.random.randn() * 2 + 20
                            }
                    
                    await websocket.send_json(update_data)
                    await asyncio.sleep(1)
                    
            elif data.get("type") == "query_point":
                lat = data.get("latitude")
                lon = data.get("longitude")
                
                point_data = {
                    "type": "point_data",
                    "location": {"latitude": lat, "longitude": lon},
                    "values": {
                        "temperature": 20 + 15 * np.cos(np.radians(lat)),
                        "wind_speed": abs(10 * np.sin(np.radians(lon))),
                        "wind_direction": (np.arctan2(np.cos(np.radians(lon)), 
                                         np.sin(np.radians(lon))) * 180 / np.pi) % 360
                    }
                }
                
                await websocket.send_json(point_data)
                
    except WebSocketDisconnect:
        print(f"Client disconnected from tileset {tileset_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/download-recipe/{tileset_id}")
async def download_recipe(tileset_id: str):
    """Download the Mapbox recipe JSON"""
    recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
    if recipe_path.exists():
        return FileResponse(
            str(recipe_path),
            filename=f"recipe_{tileset_id}.json",
            media_type="application/json"
        )
    return JSONResponse({"message": "Recipe not found."}, status_code=404)

@app.get("/download-tif/{variable}")
async def download_tif(variable: str):
    """Download a specific variable as GeoTIFF"""
    tif_path = TEMP_DIR / f"{variable}.tif"
    if tif_path.exists():
        return FileResponse(
            str(tif_path),
            filename=f"{variable}.tif",
            media_type="image/tiff"
        )
    return JSONResponse({"message": "GeoTIFF not found."}, status_code=404)

@app.get("/preview/{variable}")
async def get_preview(variable: str):
    """Get preview image for a variable"""
    # First try to find a generated preview PNG
    preview_path = TEMP_DIR / f"{variable}_preview.png"
    if preview_path.exists():
        return FileResponse(
            str(preview_path),
            media_type="image/png"
        )
    
    # If not found, try to generate from TIF
    tif_path = TEMP_DIR / f"{variable}.tif"
    if tif_path.exists():
        try:
            # Open the TIF and create a preview
            with rasterio.open(tif_path) as src:
                data = src.read(1)  # Read first band
                
                # Handle NaN values
                data = np.nan_to_num(data, nan=0.0)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Normalize data for better visualization
                valid_data = data[data != -9999]  # Exclude fill values
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                else:
                    vmin, vmax = data.min(), data.max()
                
                # Create the plot
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(variable, rotation=270, labelpad=20)
                
                # Add title
                ax.set_title(f'{variable} Visualization', fontsize=14, fontweight='bold')
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Remove axis labels
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Save to bytes
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Return as response
                buffer.seek(0)
                return Response(content=buffer.getvalue(), media_type="image/png")
                
        except Exception as e:
            print(f"Error generating preview from TIF: {e}")
    
    # If still not found, try to find the original prepared NetCDF and generate
    nc_files = list(TEMP_DIR.glob("prepared_*.nc"))
    if nc_files:
        try:
            # Use the most recent prepared file
            nc_file = max(nc_files, key=lambda p: p.stat().st_mtime)
            ds = xr.open_dataset(nc_file)
            
            if variable in ds.data_vars:
                preview_base64 = create_preview_from_variable(ds, variable)
                if preview_base64:
                    # Convert base64 back to bytes and return
                    img_data = base64.b64decode(preview_base64)
                    return Response(content=img_data, media_type="image/png")
            
            ds.close()
        except Exception as e:
            print(f"Error generating preview from NetCDF: {e}")
    
    return JSONResponse({"message": "Preview not found."}, status_code=404)

@app.get("/check-job/{job_id}")
async def check_job(job_id: str, username: str = None, token: str = None):
    """Check the status of a Mapbox tileset publish job"""
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mapbox_configured": bool(MAPBOX_TOKEN and MAPBOX_USERNAME),
        "temp_dir": str(TEMP_DIR),
        "temp_dir_exists": TEMP_DIR.exists(),
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK),
        "version": "2.0.0"
    }

# Add this endpoint to your app.py file
@app.get("/test-token", response_class=HTMLResponse)
async def token_tester_page(request: Request):
    return templates.TemplateResponse("token_tester.html", {
        "request": request,
        "MAPBOX_TOKEN": MAPBOX_TOKEN,
        "MAPBOX_USERNAME": MAPBOX_USERNAME
    })

# Also add a simpler redirect endpoint
@app.get("/test-mapbox-token")
async def redirect_to_token_tester():
    """Redirect to the token tester with pre-filled values"""
    token = MAPBOX_TOKEN or ""
    username = MAPBOX_USERNAME or ""
    
    # Redirect to the tester page with URL parameters
    return RedirectResponse(
        url=f"/test-token?token={token}&username={username}",
        status_code=302
    )

# Cleanup old temp files on startup
def cleanup_old_files():
    """Remove temp files older than 1 hour"""
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