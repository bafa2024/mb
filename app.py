from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import uuid
import platform
import shutil

app = FastAPI()

# Create necessary directories if they don't exist
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load Mapbox credentials from environment variables with better error handling
MAPBOX_TOKEN = os.getenv("MAPBOX_SECRET_TOKEN", os.getenv("MAPBOX_TOKEN"))
MAPBOX_PUBLIC_TOKEN = os.getenv("MAPBOX_PUBLIC_TOKEN", MAPBOX_TOKEN)
MAPBOX_USERNAME = os.getenv("MAPBOX_USERNAME")

# Print credential status for debugging
print("=" * 50)
print("MAPBOX CREDENTIAL STATUS:")
print(f"Token loaded: {'Yes' if MAPBOX_TOKEN else 'No'}")
print(f"Token preview: {MAPBOX_TOKEN[:20] + '...' if MAPBOX_TOKEN else 'Not set'}")
print(f"Username: {MAPBOX_USERNAME if MAPBOX_USERNAME else 'Not set'}")
print("=" * 50)

# Enhanced temp directory setup with fallbacks
def setup_temp_directory():
    """Setup temporary directory with multiple fallback options"""
    temp_options = []
    
    # Option 1: Local temp_files directory
    local_temp = Path("temp_files")
    temp_options.append(local_temp)
    
    # Option 2: System temp directory
    system_temp = Path(tempfile.gettempdir()) / "mapbox_netcdf" / str(uuid.uuid4())[:8]
    temp_options.append(system_temp)
    
    # Option 3: User home directory
    home_temp = Path.home() / ".mapbox_netcdf_temp"
    temp_options.append(home_temp)
    
    for temp_dir in temp_options:
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = temp_dir / "test_write.txt"
            test_file.write_text("test")
            test_file.unlink()
            print(f"✓ Using temp directory: {temp_dir}")
            return temp_dir
        except Exception as e:
            print(f"✗ Cannot use {temp_dir}: {e}")
            continue
    
    raise RuntimeError("Could not create a writable temporary directory")

TEMP_DIR = setup_temp_directory()

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
        
        ds.close()
        return metadata
        
    except Exception as e:
        print(f"Error preparing NetCDF: {str(e)}")
        raise

def create_preview_from_tif(tif_path: str, variable_name: str, colormap: str = 'viridis') -> str:
    """
    Create a PNG preview from a GeoTIFF file and return as base64
    """
    try:
        # Open the raster with rioxarray
        da = rioxarray.open_rasterio(tif_path)
        
        # Get the data
        if len(da.shape) == 3:
            data = da[0].values  # Take first band
        else:
            data = da.values
        
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize data for better visualization
        if data.size > 0 and not np.all(data == 0):
            vmin, vmax = np.nanpercentile(data[data != 0], [2, 98]) if np.any(data != 0) else (0, 1)
        else:
            vmin, vmax = 0, 1
        
        # Create the plot
        im = ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(variable_name, rotation=270, labelpad=20)
        
        # Add title
        ax.set_title(f'{variable_name} Visualization', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Remove axis labels for cleaner look
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
        print(f"Error creating preview from TIF: {e}")
        traceback.print_exc()
        return None

def create_animated_gif(nc_path: str, variable: str, output_path: str, 
                       colormap: str = 'viridis', fps: int = 2) -> bool:
    """
    Create an animated GIF from a NetCDF variable with time dimension
    """
    try:
        ds = xr.open_dataset(nc_path)
        
        if variable not in ds.data_vars:
            print(f"Variable {variable} not found in dataset")
            return False
        
        da = ds[variable]
        
        # Check if time dimension exists
        if 'time' not in da.dims:
            print(f"No time dimension found for {variable}")
            return False
        
        # Create frames
        frames = []
        time_steps = len(da.time)
        
        # Determine global min/max for consistent colormap
        data_values = da.values
        data_values = np.nan_to_num(data_values, nan=0.0)
        
        if data_values.size > 0 and not np.all(data_values == 0):
            vmin, vmax = np.nanpercentile(data_values[data_values != 0], [2, 98])
        else:
            vmin, vmax = 0, 1
        
        for t in range(min(time_steps, 24)):  # Limit to 24 frames for file size
            # Get data for this time step
            data = da.isel(time=t).values
            data = np.nan_to_num(data, nan=0.0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot data
            im = ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax, aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(variable, rotation=270, labelpad=20)
            
            # Add title with timestamp
            time_str = str(da.time[t].values)[:19]  # Format timestamp
            ax.set_title(f'{variable} - {time_str}', fontsize=12)
            
            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Convert to PIL Image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            frame = Image.open(buffer)
            frames.append(frame)
            plt.close()
        
        # Save as GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,  # Duration in milliseconds
                loop=0
            )
            return True
        
        ds.close()
        return False
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        traceback.print_exc()
        return False

def generate_all_previews(prepared_path: str, metadata: Dict) -> Dict:
    """
    Generate preview images for all variables
    """
    preview_data = {}
    
    try:
        ds = xr.open_dataset(prepared_path)
        
        # Define colormaps for different variable types
        colormaps = {
            'temperature': 'RdYlBu_r',
            'temp': 'RdYlBu_r',
            't2m': 'RdYlBu_r',
            'pressure': 'viridis',
            'humidity': 'Blues',
            'precipitation': 'Blues',
            'wind': 'plasma',
            'u10': 'plasma',
            'v10': 'plasma'
        }
        
        for var in metadata["variables"][:6]:  # Limit to first 6 variables
            if var in ds.data_vars:
                da = ds[var]
                
                # Handle time dimension
                if 'time' in da.dims:
                    da = da.isel(time=0)
                
                # Determine colormap
                cmap = 'viridis'
                for key, cm in colormaps.items():
                    if key in var.lower():
                        cmap = cm
                        break
                
                # Create TIF file
                session_id = str(uuid.uuid4())[:8]
                out_tif = TEMP_DIR / f"{session_id}_{var}.tif"
                try:
                    da.rio.to_raster(str(out_tif))
                    
                    # Create PNG preview
                    preview_base64 = create_preview_from_tif(str(out_tif), var, cmap)
                    
                    if preview_base64:
                        preview_data[var] = {
                            'preview': preview_base64,
                            'has_tif': True,
                            'tif_path': str(out_tif),
                            'colormap': cmap,
                            'stats': {
                                'min': float(np.nanmin(da.values)),
                                'max': float(np.nanmax(da.values)),
                                'mean': float(np.nanmean(da.values))
                            }
                        }
                        
                        # Check if we can create animated GIF
                        if 'time' in ds[var].dims and len(ds[var].time) > 1:
                            gif_path = TEMP_DIR / f"{session_id}_{var}_animated.gif"
                            if create_animated_gif(prepared_path, var, str(gif_path), cmap):
                                # Convert GIF to base64
                                with open(gif_path, 'rb') as f:
                                    gif_base64 = base64.b64encode(f.read()).decode()
                                preview_data[var]['animated_gif'] = gif_base64
                                preview_data[var]['gif_path'] = str(gif_path)
                                preview_data[var]['has_animation'] = True
                        
                except Exception as e:
                    print(f"Warning: Could not create preview for {var}: {e}")
                    traceback.print_exc()
        
        ds.close()
        return preview_data
        
    except Exception as e:
        print(f"Error generating previews: {e}")
        traceback.print_exc()
        return {}

def upload_to_mapbox_mts(token: str, username: str, nc_path: str, tileset_id: str, recipe: Dict) -> Dict:
    """
    Upload NetCDF to Mapbox using MTS API with recipe.
    """
    try:
        # Validate credentials
        if not token or not username:
            raise ValueError("Mapbox token and username are required")
        
        # Step 1: Create tileset source
        source_id = f"{tileset_id}_source"
        
        # Get S3 credentials for source upload
        cred_url = f"https://api.mapbox.com/tilesets/v1/sources/{username}/{source_id}/upload-credentials?access_token={token}"
        
        print(f"Getting upload credentials from: {cred_url}")
        cred_resp = requests.post(cred_url)
        
        if cred_resp.status_code != 200:
            error_msg = f"Failed to get upload credentials: {cred_resp.status_code} - {cred_resp.text}"
            print(error_msg)
            if cred_resp.status_code == 401:
                error_msg += "\n\nPlease check:\n1. Your Mapbox token is valid\n2. Token has 'tilesets:write' scope\n3. Username is correct"
            raise Exception(error_msg)
        
        creds = cred_resp.json()
        
        # Upload to S3
        print("Uploading to S3...")
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
        
        print("Upload to S3 completed")
        
        # Step 2: Create tileset with recipe
        tileset_url = f"https://api.mapbox.com/tilesets/v1/{username}.{tileset_id}?access_token={token}"
        tileset_data = {
            "recipe": recipe,
            "name": f"Weather Data - {tileset_id}",
            "description": "Multi-variable weather data with vector and scalar fields"
        }
        
        print(f"Creating tileset: {username}.{tileset_id}")
        create_resp = requests.post(tileset_url, json=tileset_data)
        
        if create_resp.status_code not in [200, 201]:
            # Try updating if already exists
            if create_resp.status_code == 409:
                print("Tileset exists, updating...")
                update_resp = requests.patch(tileset_url, json={"recipe": recipe})
                if update_resp.status_code != 200:
                    raise Exception(f"Failed to update tileset: {update_resp.text}")
            else:
                raise Exception(f"Failed to create tileset: {create_resp.text}")
        
        # Step 3: Publish tileset
        publish_url = f"https://api.mapbox.com/tilesets/v1/{username}.{tileset_id}/publish?access_token={token}"
        print("Publishing tileset...")
        publish_resp = requests.post(publish_url)
        
        if publish_resp.status_code != 200:
            raise Exception(f"Failed to publish tileset: {publish_resp.text}")
        
        job_id = publish_resp.json().get('jobId')
        print(f"Publishing started with job ID: {job_id}")
        
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
    upload_to_mapbox: bool = Form(True),
    create_recipe: bool = Form(True),
    generate_previews: bool = Form(True)
):
    """
    Process NetCDF file with enhanced error handling
    """
    filepath = None
    prepared_path = None
    session_id = str(uuid.uuid4())[:8]
    
    try:
        # Use environment variables if not provided
        token = mapbox_token or MAPBOX_TOKEN
        username = mapbox_username or MAPBOX_USERNAME
        
        # Debug info
        print(f"\n{'='*50}")
        print(f"Processing file: {file.filename}")
        print(f"Session ID: {session_id}")
        print(f"Temp directory: {TEMP_DIR}")
        print(f"Mapbox upload: {upload_to_mapbox}")
        print(f"Credentials available: {bool(token and username)}")
        print(f"{'='*50}\n")
        
        # Create safe filename
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        filepath = TEMP_DIR / f"{session_id}_{safe_filename}"
        
        # Save uploaded file with proper error handling
        print(f"Saving file to: {filepath}")
        try:
            content = await file.read()
            with open(filepath, "wb") as f:
                f.write(content)
                f.flush()
                if platform.system() != 'Windows':
                    os.fsync(f.fileno())
            print(f"File saved successfully, size: {len(content)} bytes")
        except Exception as e:
            print(f"Error saving file: {e}")
            traceback.print_exc()
            raise Exception(f"Failed to save uploaded file: {str(e)}")
        
        # Prepare NetCDF
        prepared_filename = f"{session_id}_prepared_{safe_filename}"
        prepared_path = TEMP_DIR / prepared_filename
        
        print(f"Preparing NetCDF file...")
        metadata = prepare_netcdf_for_mapbox(str(filepath), str(prepared_path))
        print(f"NetCDF prepared with {len(metadata['variables'])} variables")
        
        # Create response
        response_data = {
            "success": True,
            "metadata": metadata,
            "message": f"Successfully processed {file.filename}"
        }
        
        # Generate previews if requested
        if generate_previews:
            print("Generating previews...")
            preview_data = generate_all_previews(str(prepared_path), metadata)
            response_data["previews"] = preview_data
            response_data["has_previews"] = len(preview_data) > 0
            print(f"Generated {len(preview_data)} previews")
        
        # Generate GeoTIFF for each variable (for download)
        ds = xr.open_dataset(str(prepared_path))
        generated_files = []
        
        for var in metadata["variables"][:6]:  # Limit to first 6
            if var in ds.data_vars:
                da = ds[var]
                if 'time' in da.dims:
                    da = da.isel(time=0)
                
                out_tif = TEMP_DIR / f"{session_id}_{var}.tif"
                if not out_tif.exists():
                    try:
                        da.rio.to_raster(str(out_tif))
                        generated_files.append({
                            'name': var,
                            'path': str(out_tif)
                        })
                    except Exception as e:
                        print(f"Warning: Could not create GeoTIFF for {var}: {e}")
        
        ds.close()
        
        response_data["generated_files"] = [f['name'] for f in generated_files]
        response_data["session_id"] = session_id
        
        # Upload to Mapbox if requested and credentials available
        if upload_to_mapbox and create_recipe:
            if not token or not username:
                response_data["message"] += "<br><br>⚠️ Mapbox upload skipped: No credentials configured.<br>To enable upload, set MAPBOX_TOKEN and MAPBOX_USERNAME in your .env file."
                response_data["mapbox_upload"] = False
            else:
                try:
                    # Create tileset ID
                    tileset_id = re.sub(r'[^a-zA-Z0-9_-]', '_', 
                                      os.path.splitext(file.filename)[0].lower())[:32]
                    tileset_id = f"{session_id}_{tileset_id}"  # Make unique
                    
                    print(f"Creating recipe for tileset: {tileset_id}")
                    # Create recipe
                    recipe = create_recipe_for_netcdf(str(prepared_path), tileset_id, username)
                    
                    print("Uploading to Mapbox...")
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
                    recipe_path = TEMP_DIR / f"{session_id}_recipe_{tileset_id}.json"
                    with open(recipe_path, 'w') as f:
                        json.dump(recipe, f, indent=2)
                    
                    response_data["recipe_download"] = f"/download-recipe/{session_id}/{tileset_id}"
                    response_data["visualization_url"] = f"/visualize/{tileset_id}"
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Mapbox upload error: {error_msg}")
                    traceback.print_exc()
                    
                    if "401" in error_msg or "Unauthorized" in error_msg:
                        response_data["message"] += f"""<br><br>⚠️ Mapbox authentication failed.<br>
                        Please check:
                        <br>1. Your token is valid
                        <br>2. Token has 'tilesets:write' scope
                        <br>3. Username is correct"""
                    else:
                        response_data["message"] += f"<br><br>⚠️ Mapbox upload failed: {error_msg}"
                    
                    response_data["mapbox_error"] = error_msg
                    response_data["mapbox_upload"] = False
        
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error processing file: {str(e)}"
        }, status_code=500)
    
    finally:
        # Clean up old files (older than 1 hour)
        try:
            cleanup_old_files(max_age_hours=1)
        except:
            pass

@app.get("/download-tif/{session_id}/{variable}")
async def download_tif(session_id: str, variable: str):
    """Download a specific variable as GeoTIFF."""
    # Try multiple possible filenames
    possible_files = [
        TEMP_DIR / f"{session_id}_{variable}.tif",
        TEMP_DIR / f"{variable}.tif"
    ]
    
    for tif_path in possible_files:
        if tif_path.exists():
            return FileResponse(
                str(tif_path),
                filename=f"{variable}.tif",
                media_type="image/tiff"
            )
    
    return JSONResponse({"message": "GeoTIFF not found."}, status_code=404)

@app.get("/download-gif/{session_id}/{variable}")
async def download_gif(session_id: str, variable: str):
    """Download animated GIF for a variable."""
    gif_path = TEMP_DIR / f"{session_id}_{variable}_animated.gif"
    if gif_path.exists():
        return FileResponse(
            str(gif_path),
            filename=f"{variable}_animated.gif",
            media_type="image/gif"
        )
    return JSONResponse({"message": "Animated GIF not found."}, status_code=404)

@app.get("/download-recipe/{session_id}/{tileset_id}")
async def download_recipe(session_id: str, tileset_id: str):
    """Download the Mapbox recipe JSON."""
    recipe_path = TEMP_DIR / f"{session_id}_recipe_{tileset_id}.json"
    if recipe_path.exists():
        return FileResponse(
            str(recipe_path),
            filename=f"recipe_{tileset_id}.json",
            media_type="application/json"
        )
    return JSONResponse({"message": "Recipe not found."}, status_code=404)

@app.get("/preview/{session_id}/{variable}")
async def get_preview(session_id: str, variable: str):
    """Get preview image for a variable."""
    # Try to find TIF and generate preview
    tif_path = TEMP_DIR / f"{session_id}_{variable}.tif"
    if tif_path.exists():
        preview_base64 = create_preview_from_tif(str(tif_path), variable)
        if preview_base64:
            img_data = base64.b64decode(preview_base64)
            return Response(content=img_data, media_type="image/png")
    
    return JSONResponse({"message": "Preview not found."}, status_code=404)

@app.get("/visualize/{tileset_id}")
async def visualize_tileset(request: Request, tileset_id: str):
    """
    Display multi-variable visualization page for a tileset
    """
    # Look for recipe file
    recipe = {}
    for file in TEMP_DIR.glob(f"*_recipe_{tileset_id}.json"):
        try:
            with open(file, 'r') as f:
                recipe = json.load(f)
                break
        except:
            pass
    
    # Use public token for visualization
    public_token = MAPBOX_PUBLIC_TOKEN or MAPBOX_TOKEN
    
    return templates.TemplateResponse("multi_variable_visualization.html", {
        "request": request,
        "tileset_id": f"{MAPBOX_USERNAME}.{tileset_id}" if MAPBOX_USERNAME else tileset_id,
        "mapbox_token": public_token,
        "recipe": recipe
    })

@app.get("/check-job/{job_id}")
async def check_job(job_id: str, username: str = None, token: str = None):
    """Check the status of a Mapbox tileset publish job."""
    try:
        username = username or MAPBOX_USERNAME
        token = token or MAPBOX_TOKEN
        
        if not username or not token:
            return JSONResponse({"error": "Mapbox credentials not configured"}, status_code=400)
        
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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mapbox_configured": bool(MAPBOX_TOKEN and MAPBOX_USERNAME),
        "temp_dir": str(TEMP_DIR),
        "temp_dir_exists": TEMP_DIR.exists(),
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK),
        "credentials": {
            "token_set": bool(MAPBOX_TOKEN),
            "username_set": bool(MAPBOX_USERNAME)
        }
    }

# Cleanup old temp files
def cleanup_old_files(max_age_hours: int = 1):
    """Remove temp files older than specified hours."""
    try:
        current_time = time.time()
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
                if file_age_hours > max_age_hours:
                    try:
                        file_path.unlink()
                        print(f"Cleaned up old file: {file_path.name}")
                    except:
                        pass
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Run cleanup on startup
cleanup_old_files(max_age_hours=24)  # Clean files older than 24 hours on startup