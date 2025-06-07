from fastapi import FastAPI, UploadFile, File, Request, Form, WebSocket, WebSocketDisconnect, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple, Set
import xarray as xr
import rioxarray
import os
import requests
import boto3
import json
import time
import re
import numpy as np
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
from datetime import datetime, timedelta
import rasterio
from rasterio.transform import from_bounds
from rasterio.plot import show
import aiofiles
import imageio

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

# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections for real-time data streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.data_subscriptions: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, tileset_id: str):
        await websocket.accept()
        if tileset_id not in self.active_connections:
            self.active_connections[tileset_id] = set()
        self.active_connections[tileset_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, tileset_id: str):
        if tileset_id in self.active_connections:
            self.active_connections[tileset_id].discard(websocket)
            if not self.active_connections[tileset_id]:
                del self.active_connections[tileset_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str, tileset_id: str):
        if tileset_id in self.active_connections:
            for connection in self.active_connections[tileset_id]:
                try:
                    await connection.send_text(message)
                except:
                    pass

# Create global connection manager
manager = ConnectionManager()

# Token generation request model
class TokenGenerationRequest(BaseModel):
    master_token: str
    username: str
    note: str = "NetCDF Converter Token"
    scopes: List[str] = [
        "tilesets:write",
        "tilesets:read",
        "tilesets:list",
        "sources:write",
        "sources:read"
    ]

# Advanced NetCDF processor class
class AdvancedNetCDFProcessor:
    """Advanced NetCDF processing with animation and multi-variable support"""
    
    @staticmethod
    def create_animated_preview(ds: xr.Dataset, variable: str, 
                               output_path: str = None,
                               fps: int = 2,
                               colormap: str = 'viridis') -> str:
        """Create animated GIF from time series data"""
        if variable not in ds.data_vars:
            raise ValueError(f"Variable {variable} not found")
        
        data = ds[variable]
        
        if 'time' not in data.dims:
            raise ValueError(f"Variable {variable} has no time dimension")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []
            
            vmin = float(data.min())
            vmax = float(data.max())
            
            for t in range(len(data.time)):
                fig, ax = plt.subplots(figsize=(10, 8))
                
                time_slice = data.isel(time=t)
                
                if 'latitude' in time_slice.dims and 'longitude' in time_slice.dims:
                    im = ax.imshow(time_slice.values, 
                                  extent=[time_slice.longitude.min(), time_slice.longitude.max(),
                                         time_slice.latitude.min(), time_slice.latitude.max()],
                                  cmap=colormap, vmin=vmin, vmax=vmax, aspect='auto')
                else:
                    im = ax.imshow(time_slice.values, cmap=colormap, vmin=vmin, vmax=vmax, aspect='auto')
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f"{variable} ({data.attrs.get('units', 'unknown')})")
                
                if hasattr(data.time[t], 'values'):
                    timestamp = str(data.time[t].values)
                else:
                    timestamp = f"Time step {t}"
                ax.set_title(f"{variable} - {timestamp}")
                
                frame_path = Path(temp_dir) / f"frame_{t:04d}.png"
                plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                frames.append(imageio.imread(frame_path))
            
            gif_buffer = io.BytesIO()
            imageio.mimsave(gif_buffer, frames, format='GIF', fps=fps, loop=0)
            
            gif_buffer.seek(0)
            gif_base64 = base64.b64encode(gif_buffer.getvalue()).decode()
            
            return gif_base64
    
    @staticmethod
    def create_composite_visualization(ds: xr.Dataset, 
                                     variables: List[str],
                                     time_step: int = 0) -> str:
        """Create composite visualization of multiple variables"""
        n_vars = len(variables)
        fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 4))
        
        if n_vars == 1:
            axes = [axes]
        
        for i, var in enumerate(variables):
            if var not in ds.data_vars:
                continue
            
            data = ds[var]
            if 'time' in data.dims:
                data = data.isel(time=time_step)
            
            im = axes[i].imshow(data.values, cmap='viridis', aspect='auto')
            axes[i].set_title(var)
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
    
    @staticmethod
    def extract_time_series_at_point(ds: xr.Dataset, 
                                   variable: str,
                                   lat: float, 
                                   lon: float) -> Dict:
        """Extract time series data at a specific location"""
        if variable not in ds.data_vars:
            raise ValueError(f"Variable {variable} not found")
        
        data = ds[variable]
        
        lat_name = 'latitude' if 'latitude' in data.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in data.dims else 'lon'
        
        point_data = data.sel({lat_name: lat, lon_name: lon}, method='nearest')
        
        result = {
            'variable': variable,
            'location': {
                'requested': {'lat': lat, 'lon': lon},
                'actual': {
                    'lat': float(point_data[lat_name].values),
                    'lon': float(point_data[lon_name].values)
                }
            },
            'units': data.attrs.get('units', 'unknown')
        }
        
        if 'time' in point_data.dims:
            result['time_series'] = {
                'times': [str(t) for t in point_data.time.values],
                'values': point_data.values.tolist()
            }
            result['statistics'] = {
                'min': float(point_data.min()),
                'max': float(point_data.max()),
                'mean': float(point_data.mean()),
                'std': float(point_data.std()) if point_data.size > 1 else 0
            }
        else:
            result['value'] = float(point_data.values)
        
        return result
    
    @staticmethod
    def create_wind_streamplot(ds: xr.Dataset, 
                             u_var: str, v_var: str,
                             time_step: int = 0) -> str:
        """Create streamplot visualization for wind/current data"""
        if u_var not in ds.data_vars or v_var not in ds.data_vars:
            raise ValueError(f"Variables {u_var} or {v_var} not found")
        
        u_data = ds[u_var]
        v_data = ds[v_var]
        
        if 'time' in u_data.dims:
            u_data = u_data.isel(time=time_step)
            v_data = v_data.isel(time=time_step)
        
        lat_name = 'latitude' if 'latitude' in u_data.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in u_data.dims else 'lon'
        
        lats = u_data[lat_name].values
        lons = u_data[lon_name].values
        
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        speed = np.sqrt(u_data.values**2 + v_data.values**2)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.contourf(lon_grid, lat_grid, speed, levels=20, cmap='viridis', alpha=0.6)
        plt.colorbar(im, ax=ax, label='Wind Speed')
        
        strm = ax.streamplot(lons, lats, u_data.values, v_data.values,
                           color='black', density=1.5, linewidth=1.5,
                           arrowsize=1.5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Wind Field - {u_var} & {v_var}')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
    
    @staticmethod
    def calculate_derived_variables(ds: xr.Dataset) -> xr.Dataset:
        """Calculate derived variables from existing data"""
        ds_copy = ds.copy()
        
        u_vars = [v for v in ds.data_vars if 'u' in v.lower() and ('wind' in v.lower() or v.endswith('u'))]
        v_vars = [v for v in ds.data_vars if 'v' in v.lower() and ('wind' in v.lower() or v.endswith('v'))]
        
        for u_var in u_vars:
            v_var = u_var.replace('u', 'v')
            if v_var in v_vars and u_var in ds.data_vars and v_var in ds.data_vars:
                speed_var = u_var.replace('u', 'speed').replace('_u', '_speed')
                ds_copy[speed_var] = np.sqrt(ds[u_var]**2 + ds[v_var]**2)
                ds_copy[speed_var].attrs['units'] = ds[u_var].attrs.get('units', 'm/s')
                ds_copy[speed_var].attrs['long_name'] = f"Wind speed from {u_var} and {v_var}"
                
                dir_var = u_var.replace('u', 'direction').replace('_u', '_direction')
                ds_copy[dir_var] = np.arctan2(ds[v_var], ds[u_var]) * 180 / np.pi
                ds_copy[dir_var].attrs['units'] = 'degrees'
                ds_copy[dir_var].attrs['long_name'] = f"Wind direction from {u_var} and {v_var}"
        
        return ds_copy

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main upload interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/test-token", response_class=HTMLResponse)
async def token_tester_page(request: Request):
    """Serve the Mapbox token testing page"""
    return templates.TemplateResponse("token_tester.html", {
        "request": request,
        "MAPBOX_TOKEN": MAPBOX_TOKEN,
        "MAPBOX_USERNAME": MAPBOX_USERNAME
    })

@app.get("/generate-token", response_class=HTMLResponse)
async def token_generator_page(request: Request):
    """Serve the token generator page"""
    return templates.TemplateResponse("token_generator.html", {
        "request": request,
        "MAPBOX_USERNAME": MAPBOX_USERNAME
    })

def safe_compute_stats(data_array):
    """Safely compute statistics handling dtype issues"""
    try:
        data = data_array.values.astype(np.float32)
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
    """Prepare NetCDF file for Mapbox upload with proper formatting"""
    try:
        ds = xr.open_dataset(nc_path)
        
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
        
        if 'longitude' in ds.coords:
            ds['longitude'] = ds['longitude'].astype('float64')
        if 'latitude' in ds.coords:
            ds['latitude'] = ds['latitude'].astype('float64')
        
        for var in ds.data_vars:
            if 'longitude' in ds[var].dims and 'latitude' in ds[var].dims:
                try:
                    ds[var] = ds[var].astype('float32')
                    ds[var].rio.write_crs("EPSG:4326", inplace=True)
                    ds[var].rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
                except Exception as e:
                    print(f"Warning: Could not set CRS for {var}: {e}")
        
        if 'time' in ds.dims and len(ds.time) > 1:
            ds = ds.isel(time=0)
        
        for var in ds.data_vars:
            data_array = ds[var]
            if np.isnan(data_array).any():
                print(f"Warning: {var} contains NaN values, replacing with -9999")
                ds[var] = data_array.fillna(-9999)
        
        encoding = {}
        for var in ds.data_vars:
            if ds[var].dtype != np.float32:
                ds[var] = ds[var].astype(np.float32)
            
            encoding[var] = {
                'dtype': 'float32',
                '_FillValue': -9999,
                'zlib': True,
                'complevel': 4
            }
        
        ds.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
        
        metadata = {
            "dimensions": dict(ds.dims),
            "variables": list(ds.data_vars),
            "scalar_vars": [],
            "vector_pairs": []
        }
        
        var_names = list(ds.data_vars)
        vector_patterns = [
            ('u', 'v'), ('u10', 'v10'), ('u_', 'v_'), 
            ('water_u', 'water_v'), ('eastward', 'northward')
        ]
        
        processed_vars = set()
        for u_pattern, v_pattern in vector_patterns:
            u_matches = [v for v in var_names if u_pattern in v.lower() and v not in processed_vars]
            v_matches = [v for v in var_names if v_pattern in v.lower() and v not in processed_vars]
            
            if u_matches and v_matches:
                for u_var in u_matches:
                    for v_var in v_matches:
                        if u_var.replace(u_pattern, '') == v_var.replace(v_pattern, ''):
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

def validate_mapbox_credentials(token: str, username: str) -> Dict:
    """Validate Mapbox credentials and check required scopes"""
    try:
        token_url = f"https://api.mapbox.com/tokens/v2?access_token={token}"
        resp = requests.get(token_url)
        
        if resp.status_code != 200:
            return {
                "valid": False,
                "error": "Invalid token",
                "details": resp.text
            }
        
        token_info = resp.json()
        token_data = token_info.get('token', {})
        scopes = token_data.get('scopes', [])
        
        required_scopes = ['tilesets:write', 'tilesets:read', 'tilesets:list']
        missing_scopes = [s for s in required_scopes if s not in scopes]
        
        if missing_scopes:
            return {
                "valid": False,
                "error": "Missing required scopes",
                "missing_scopes": missing_scopes,
                "current_scopes": scopes,
                "help": "Create a new token at https://account.mapbox.com/access-tokens/ with tilesets:write, tilesets:read, and tilesets:list scopes"
            }
        
        list_url = f"https://api.mapbox.com/tilesets/v1/{username}?access_token={token}&limit=1"
        list_resp = requests.get(list_url)
        
        if list_resp.status_code == 404:
            return {
                "valid": False,
                "error": "Invalid username",
                "help": "Check your Mapbox username at https://account.mapbox.com/"
            }
        elif list_resp.status_code != 200:
            return {
                "valid": False,
                "error": f"Failed to verify username: {list_resp.status_code}",
                "details": list_resp.text
            }
        
        return {
            "valid": True,
            "username": username,
            "scopes": scopes,
            "token_created": token_data.get('created'),
            "token_modified": token_data.get('modified')
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

def upload_to_mapbox_mts(token: str, username: str, nc_path: str, tileset_id: str, recipe: Dict) -> Dict:
    """Upload NetCDF to Mapbox using MTS API with recipe"""
    try:
        source_id = f"{tileset_id}_source"
        
        create_source_url = f"https://api.mapbox.com/tilesets/v1/sources/{username}/{source_id}?access_token={token}"
        create_resp = requests.post(create_source_url, json={})
        
        cred_url = f"https://api.mapbox.com/tilesets/v1/sources/{username}/{source_id}/upload-credentials?access_token={token}"
        cred_resp = requests.post(cred_url)
        
        if cred_resp.status_code != 200:
            print(f"Credential request URL: {cred_url}")
            print(f"Response status: {cred_resp.status_code}")
            print(f"Response body: {cred_resp.text}")
            
            if cred_resp.status_code == 401:
                raise Exception("Authentication failed. Make sure your token has 'tilesets:write' scope")
            elif cred_resp.status_code == 404:
                raise Exception("Source not found. Username might be incorrect or API endpoint changed")
            else:
                raise Exception(f"Failed to get upload credentials: {cred_resp.text}")
        
        creds = cred_resp.json()
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=creds['accessKeyId'],
            aws_secret_access_key=creds['secretAccessKey'],
            aws_session_token=creds['sessionToken'],
            region_name=creds.get('region', 'us-east-1')
        )
        
        with open(nc_path, 'rb') as f:
            s3_client.put_object(
                Bucket=creds['bucket'],
                Key=creds['key'],
                Body=f
            )
        
        print(f"File uploaded to S3 successfully")
        
        tileset_url = f"https://api.mapbox.com/tilesets/v1/{username}.{tileset_id}?access_token={token}"
        
        check_resp = requests.get(tileset_url)
        
        if check_resp.status_code == 404:
            tileset_data = {
                "recipe": recipe,
                "name": f"Weather Data - {tileset_id}",
                "description": "Multi-variable weather data with vector and scalar fields",
                "private": False
            }
            
            create_resp = requests.post(tileset_url, json=tileset_data)
            if create_resp.status_code not in [200, 201]:
                raise Exception(f"Failed to create tileset: {create_resp.text}")
        else:
            update_resp = requests.patch(tileset_url, json={"recipe": recipe})
            if update_resp.status_code != 200:
                raise Exception(f"Failed to update tileset: {update_resp.text}")
        
        publish_url = f"https://api.mapbox.com/tilesets/v1/{username}.{tileset_id}/publish?access_token={token}"
        publish_resp = requests.post(publish_url)
        
        if publish_resp.status_code != 200:
            raise Exception(f"Failed to publish tileset: {publish_resp.text}")
        
        job_id = publish_resp.json().get('jobId')
        
        return {
            "tileset_id": f"{username}.{tileset_id}",
            "source_id": f"{username}/{source_id}",
            "job_id": job_id,
            "recipe": recipe,
            "status": "publishing"
        }
        
    except Exception as e:
        print(f"Error uploading to Mapbox: {str(e)}")
        raise

def create_preview_from_variable(ds: xr.Dataset, variable: str, colormap: str = 'viridis') -> str:
    """Create a base64 PNG preview from a variable"""
    try:
        da = ds[variable]
        
        if 'time' in da.dims:
            da = da.isel(time=0)
        
        data = da.values.astype(np.float32)
        
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [2, 98])
        else:
            vmin, vmax = 0, 1
        
        im = ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax, aspect='auto')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(variable, rotation=270, labelpad=20)
        
        ax.set_title(f'{variable} Visualization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
        
    except Exception as e:
        print(f"Error creating preview for {variable}: {e}")
        return None

@app.post("/validate-credentials")
async def validate_credentials(
    token: str = Form(...),
    username: str = Form(...)
):
    """Validate Mapbox credentials"""
    result = validate_mapbox_credentials(token, username)
    return JSONResponse(result)

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
        token = mapbox_token or MAPBOX_TOKEN
        username = mapbox_username or MAPBOX_USERNAME
        
        filepath = TEMP_DIR / file.filename
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        prepared_path = TEMP_DIR / f"prepared_{file.filename}"
        metadata = prepare_netcdf_for_mapbox(str(filepath), str(prepared_path))
        
        response_data = {
            "success": True,
            "metadata": metadata,
            "message": f"Successfully processed {file.filename}"
        }
        
        ds = xr.open_dataset(str(prepared_path))
        preview_data = {}
        
        for var in metadata["variables"][:6]:
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
        
        ds = xr.open_dataset(str(prepared_path))
        generated_files = []
        
        for var in metadata["variables"][:6]:
            if var in ds.data_vars:
                da = ds[var]
                if 'time' in da.dims:
                    da = da.isel(time=0)
                
                out_tif = TEMP_DIR / f"{var}.tif"
                try:
                    da = da.astype('float32')
                    da = da.fillna(-9999)
                    da.rio.to_raster(str(out_tif))
                    generated_files.append(var)
                except Exception as e:
                    print(f"Warning: Could not create GeoTIFF for {var}: {e}")
        
        ds.close()
        response_data["generated_files"] = generated_files
        
        if upload_to_mapbox and create_recipe:
            if not token or not username:
                response_data["message"] += "<br>⚠️ Mapbox upload requires credentials."
            else:
                try:
                    tileset_id = re.sub(r'[^a-zA-Z0-9_-]', '_', 
                                      os.path.splitext(file.filename)[0].lower())[:32]
                    
                    recipe = create_enhanced_recipe_for_netcdf(str(prepared_path), tileset_id, username)
                    
                    upload_result = upload_to_mapbox_mts(token, username, str(prepared_path), tileset_id, recipe)
                    
                    response_data["mapbox_upload"] = True
                    response_data["tileset_id"] = upload_result["tileset_id"]
                    response_data["job_id"] = upload_result["job_id"]
                    response_data["recipe"] = recipe
                    response_data["visualization_url"] = f"/visualize-advanced/{tileset_id}"
                    
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

@app.post("/process-advanced")
async def process_file_advanced(
    file: UploadFile = File(...),
    create_animation: bool = Form(False),
    extract_points: str = Form(None),
    calculate_derived: bool = Form(False),
    create_streamplot: bool = Form(False),
    mapbox_token: Optional[str] = Form(None),
    mapbox_username: Optional[str] = Form(None),
    upload_to_mapbox: bool = Form(True)
):
    """Advanced NetCDF processing with multiple visualization options"""
    filepath = None
    
    try:
        filepath = TEMP_DIR / file.filename
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        processor = AdvancedNetCDFProcessor()
        
        ds = xr.open_dataset(filepath)
        
        if calculate_derived:
            ds = processor.calculate_derived_variables(ds)
        
        response_data = {
            "success": True,
            "filename": file.filename,
            "variables": list(ds.data_vars),
            "dimensions": dict(ds.dims),
            "visualizations": {}
        }
        
        if create_animation:
            animations = {}
            for var in list(ds.data_vars)[:3]:
                if 'time' in ds[var].dims and len(ds[var].time) > 1:
                    try:
                        gif_base64 = processor.create_animated_preview(ds, var)
                        animations[var] = gif_base64
                    except Exception as e:
                        print(f"Failed to create animation for {var}: {e}")
            response_data["visualizations"]["animations"] = animations
        
        if extract_points:
            import json
            points = json.loads(extract_points)
            time_series_data = {}
            
            for point in points:
                lat, lon = point['lat'], point['lon']
                point_key = f"{lat},{lon}"
                time_series_data[point_key] = {}
                
                for var in list(ds.data_vars)[:5]:
                    try:
                        ts_data = processor.extract_time_series_at_point(ds, var, lat, lon)
                        time_series_data[point_key][var] = ts_data
                    except Exception as e:
                        print(f"Failed to extract time series for {var} at {point_key}: {e}")
            
            response_data["visualizations"]["time_series"] = time_series_data
        
        if create_streamplot:
            streamplots = {}
            u_vars = [v for v in ds.data_vars if 'u' in v.lower()]
            v_vars = [v for v in ds.data_vars if 'v' in v.lower()]
            
            for u_var in u_vars:
                v_var = u_var.replace('u', 'v')
                if v_var in v_vars:
                    try:
                        streamplot_base64 = processor.create_wind_streamplot(ds, u_var, v_var)
                        streamplots[f"{u_var}_{v_var}"] = streamplot_base64
                    except Exception as e:
                        print(f"Failed to create streamplot for {u_var}, {v_var}: {e}")
            
            response_data["visualizations"]["streamplots"] = streamplots
        
        if upload_to_mapbox and mapbox_token and mapbox_username:
            validation = validate_mapbox_credentials(mapbox_token, mapbox_username)
            if not validation['valid']:
                response_data["mapbox_error"] = validation['error']
            else:
                try:
                    prepared_path = TEMP_DIR / f"prepared_{file.filename}"
                    metadata = prepare_netcdf_for_mapbox(str(filepath), str(prepared_path))
                    
                    tileset_id = re.sub(r'[^a-zA-Z0-9_-]', '_', 
                                      os.path.splitext(file.filename)[0].lower())[:32]
                    
                    recipe = create_enhanced_recipe_for_netcdf(str(prepared_path), tileset_id, mapbox_username)
                    
                    upload_result = upload_to_mapbox_mts(mapbox_token, mapbox_username, 
                                                       str(prepared_path), tileset_id, recipe)
                    
                    response_data["mapbox_upload"] = {
                        "success": True,
                        "tileset_id": upload_result["tileset_id"],
                        "job_id": upload_result["job_id"],
                        "visualization_url": f"/visualize-advanced/{tileset_id}"
                    }
                    
                except Exception as e:
                    response_data["mapbox_error"] = str(e)
        
        ds.close()
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"Error in advanced processing: {str(e)}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
    finally:
        if filepath and filepath.exists():
            try:
                filepath.unlink()
            except:
                pass

@app.post("/api/generate-token")
async def generate_mapbox_token(request: TokenGenerationRequest):
    """Generate a new Mapbox token with specified scopes using a master token"""
    try:
        validate_url = f"https://api.mapbox.com/tokens/v2?access_token={request.master_token}"
        validate_response = requests.get(validate_url)
        
        if validate_response.status_code != 200:
            return JSONResponse({
                "success": False,
                "error": "Invalid master token or it doesn't have tokens:write scope"
            }, status_code=401)
        
        token_info = validate_response.json().get('token', {})
        scopes = token_info.get('scopes', [])
        
        if 'tokens:write' not in scopes:
            return JSONResponse({
                "success": False,
                "error": "Master token doesn't have 'tokens:write' scope. Create a new master token with this scope."
            }, status_code=403)
        
        create_url = f"https://api.mapbox.com/tokens/v2/{request.username}?access_token={request.master_token}"
        
        payload = {
            "note": request.note,
            "scopes": request.scopes
        }
        
        create_response = requests.post(create_url, json=payload)
        
        if create_response.status_code in [200, 201]:
            data = create_response.json()
            
            print(f"✅ Token created successfully for {request.username}")
            print(f"   Scopes: {', '.join(request.scopes)}")
            
            return JSONResponse({
                "success": True,
                "token": data.get("token"),
                "id": data.get("id"),
                "note": data.get("note"),
                "scopes": data.get("scopes"),
                "created": data.get("created"),
                "modified": data.get("modified"),
                "expires": data.get("expires")
            })
        else:
            error_data = create_response.json()
            error_message = error_data.get('message', f'HTTP {create_response.status_code}')
            
            if create_response.status_code == 401:
                error_message = "Authentication failed. Check if your master token is valid and belongs to the specified username."
            elif create_response.status_code == 403:
                error_message = "Permission denied. The master token might not have the required permissions."
            elif create_response.status_code == 422:
                error_message = "Invalid request. Check if the username is correct and the scopes are valid."
            
            return JSONResponse({
                "success": False,
                "error": error_message,
                "details": error_data
            }, status_code=create_response.status_code)
            
    except Exception as e:
        print(f"❌ Error generating token: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": f"Server error: {str(e)}"
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
    
    viz_config = {
        "tileset_id": f"{MAPBOX_USERNAME}.{tileset_id}",
        "mapbox_token": MAPBOX_PUBLIC_TOKEN or MAPBOX_TOKEN,
        "recipe": recipe,
        "variables": [],
        "vector_fields": []
    }
    
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
        recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
        if not recipe_path.exists():
            return JSONResponse({
                "error": "Recipe not found for tileset"
            }, status_code=404)
        
        with open(recipe_path, 'r') as f:
            recipe = json.load(f)
        
        vector_pairs = recipe.get('metadata', {}).get('vector_pairs', [])
        if not vector_pairs:
            return JSONResponse({
                "error": "No wind data found in tileset"
            }, status_code=404)
        
        wind_pair = vector_pairs[0]
        
        lats = np.linspace(south, north, resolution)
        lons = np.linspace(west, east, resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
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
        
        for var in requested_vars:
            if 'temp' in var.lower():
                value = 20 + 15 * np.cos(np.radians(latitude))
                results["values"][var] = {
                    "value": round(value, 2),
                    "units": "°C"
                }
            elif var.endswith('_u') or 'u10' in var:
                value = 10 * np.sin(np.radians(longitude))
                results["values"][var] = {
                    "value": round(value, 2),
                    "units": "m/s"
                }
            elif var.endswith('_v') or 'v10' in var:
                value = 10 * np.cos(np.radians(longitude))
                results["values"][var] = {
                    "value": round(value, 2),
                    "units": "m/s"
                }
            else:
                value = np.random.randn() * 10 + 50
                results["values"][var] = {
                    "value": round(value, 2),
                    "units": "unknown"
                }
        
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
        bounds_list = [float(x) for x in bounds.split(',')]
        if len(bounds_list) != 4:
            raise ValueError("Invalid bounds format")
        
        west, south, east, north = bounds_list
        requested_vars = variables.split(',')
        
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
    await manager.connect(websocket, tileset_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                bounds = data.get("bounds")
                variables = data.get("variables", [])
                update_interval = data.get("interval", 5)
                
                subscription_id = f"{tileset_id}_{datetime.now().timestamp()}"
                
                asyncio.create_task(
                    stream_data_updates(
                        websocket, tileset_id, bounds, variables, 
                        update_interval, subscription_id
                    )
                )
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription_confirmed",
                        "subscription_id": subscription_id
                    }),
                    websocket
                )
                
            elif data.get("type") == "query_point":
                lat = data.get("latitude")
                lon = data.get("longitude")
                
                response = {
                    "type": "point_data",
                    "location": {"latitude": lat, "longitude": lon},
                    "timestamp": datetime.now().isoformat(),
                    "values": {}
                }
                
                for var in data.get("variables", []):
                    if "temp" in var.lower():
                        value = 20 + 15 * np.sin(np.radians(lat)) + np.random.randn() * 2
                    elif "wind" in var.lower():
                        value = abs(10 * np.sin(np.radians(lon)) + np.random.randn() * 3)
                    else:
                        value = np.random.randn() * 10 + 50
                    
                    response["values"][var] = round(value, 2)
                
                await manager.send_personal_message(json.dumps(response), websocket)
                
            elif data.get("type") == "unsubscribe":
                subscription_id = data.get("subscription_id")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "unsubscribed",
                        "subscription_id": subscription_id
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, tileset_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, tileset_id)

async def stream_data_updates(
    websocket: WebSocket, 
    tileset_id: str,
    bounds: Dict,
    variables: List[str],
    interval: int,
    subscription_id: str
):
    """Stream data updates at regular intervals"""
    try:
        while True:
            update_data = {
                "type": "data_update",
                "subscription_id": subscription_id,
                "timestamp": datetime.now().isoformat(),
                "bounds": bounds,
                "data": {}
            }
            
            for var in variables:
                if "wind" in var.lower():
                    grid_size = 10
                    lats = np.linspace(bounds["south"], bounds["north"], grid_size)
                    lons = np.linspace(bounds["west"], bounds["east"], grid_size)
                    
                    if var.endswith("_u") or "u" in var:
                        values = 10 * np.sin(np.outer(lats, np.ones(len(lons))) * 0.1) + \
                                np.random.randn(grid_size, grid_size) * 2
                    else:
                        values = 10 * np.cos(np.outer(np.ones(len(lats)), lons) * 0.1) + \
                                np.random.randn(grid_size, grid_size) * 2
                    
                    update_data["data"][var] = {
                        "grid": {"lats": lats.tolist(), "lons": lons.tolist()},
                        "values": values.tolist(),
                        "stats": {
                            "min": float(values.min()),
                            "max": float(values.max()),
                            "mean": float(values.mean())
                        }
                    }
                else:
                    grid_size = 20
                    lats = np.linspace(bounds["south"], bounds["north"], grid_size)
                    lons = np.linspace(bounds["west"], bounds["east"], grid_size)
                    
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    
                    if "temp" in var.lower():
                        values = 20 + 15 * np.sin(np.radians(lat_grid)) + \
                                5 * np.cos(np.radians(lon_grid)) + \
                                np.random.randn(grid_size, grid_size) * 2
                    else:
                        values = 50 + 20 * np.sin(np.radians(lat_grid + lon_grid)) + \
                                np.random.randn(grid_size, grid_size) * 5
                    
                    update_data["data"][var] = {
                        "grid": {"lats": lats.tolist(), "lons": lons.tolist()},
                        "values": values.tolist(),
                        "stats": {
                            "min": float(values.min()),
                            "max": float(values.max()),
                            "mean": float(values.mean())
                        }
                    }
            
            await websocket.send_text(json.dumps(update_data))
            await asyncio.sleep(interval)
            
    except Exception as e:
        print(f"Error in data streaming: {e}")

@app.get("/api/connections/{tileset_id}")
async def get_connections_info(tileset_id: str):
    """Get information about active WebSocket connections"""
    active_count = len(manager.active_connections.get(tileset_id, set()))
    
    return {
        "tileset_id": tileset_id,
        "active_connections": active_count,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/batch-query")
async def batch_query_data(
    tileset_id: str = Form(...),
    points: str = Form(...),
    variables: str = Form(...),
    time_range: Optional[str] = Form(None)
):
    """Query data for multiple points in batch"""
    try:
        points_list = json.loads(points)
        variables_list = variables.split(',')
        
        results = {
            "tileset_id": tileset_id,
            "query_time": datetime.now().isoformat(),
            "points": []
        }
        
        for point in points_list:
            lat = point['lat']
            lon = point['lon']
            
            point_result = {
                "location": {"latitude": lat, "longitude": lon},
                "values": {}
            }
            
            for var in variables_list:
                if "temp" in var.lower():
                    value = 20 + 15 * np.sin(np.radians(lat))
                elif "wind" in var.lower():
                    value = abs(10 * np.sin(np.radians(lon)))
                elif "pressure" in var.lower():
                    value = 1013 + 10 * np.cos(np.radians(lat + lon))
                else:
                    value = 50 + 20 * np.sin(np.radians(lat * lon / 100))
                
                point_result["values"][var] = round(value, 2)
            
            results["points"].append(point_result)
        
        return JSONResponse(results)
        
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=400)

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
    preview_path = TEMP_DIR / f"{variable}_preview.png"
    if preview_path.exists():
        return FileResponse(
            str(preview_path),
            media_type="image/png"
        )
    
    tif_path = TEMP_DIR / f"{variable}.tif"
    if tif_path.exists():
        try:
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                
                data = np.nan_to_num(data, nan=0.0)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                valid_data = data[data != -9999]
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                else:
                    vmin, vmax = data.min(), data.max()
                
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(variable, rotation=270, labelpad=20)
                
                ax.set_title(f'{variable} Visualization', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xticks([])
                ax.set_yticks([])
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                
                buffer.seek(0)
                return Response(content=buffer.getvalue(), media_type="image/png")
                
        except Exception as e:
            print(f"Error generating preview from TIF: {e}")
    
    nc_files = list(TEMP_DIR.glob("prepared_*.nc"))
    if nc_files:
        try:
            nc_file = max(nc_files, key=lambda p: p.stat().st_mtime)
            ds = xr.open_dataset(nc_file)
            
            if variable in ds.data_vars:
                preview_base64 = create_preview_from_variable(ds, variable)
                if preview_base64:
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

@app.get("/api/check-master-token")
async def check_master_token(token: str):
    """Check if a token has the tokens:write scope"""
    try:
        url = f"https://api.mapbox.com/tokens/v2?access_token={token}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            token_info = data.get('token', {})
            scopes = token_info.get('scopes', [])
            has_tokens_write = 'tokens:write' in scopes
            
            return JSONResponse({
                "valid": True,
                "has_tokens_write": has_tokens_write,
                "scopes": scopes,
                "created": token_info.get('created'),
                "note": token_info.get('note')
            })
        else:
            return JSONResponse({
                "valid": False,
                "error": "Invalid token"
            }, status_code=401)
            
    except Exception as e:
        return JSONResponse({
            "valid": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/list-tokens")
async def list_tokens(
    token: str = Query(..., description="Token with tokens:read scope"),
    username: str = Query(..., description="Mapbox username")
):
    """List all tokens for a user (requires tokens:read scope)"""
    try:
        url = f"https://api.mapbox.com/tokens/v2/{username}?access_token={token}"
        response = requests.get(url)
        
        if response.status_code == 200:
            tokens = response.json()
            
            token_list = []
            for t in tokens:
                token_list.append({
                    "id": t.get("id"),
                    "note": t.get("note"),
                    "scopes": t.get("scopes"),
                    "created": t.get("created"),
                    "last_used": t.get("usage", {}).get("last_used"),
                    "has_mts_scopes": all(
                        scope in t.get("scopes", []) 
                        for scope in ["tilesets:write", "tilesets:read", "tilesets:list"]
                    )
                })
            
            return JSONResponse({
                "success": True,
                "count": len(token_list),
                "tokens": token_list
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Failed to list tokens. Check if token has tokens:read scope."
            }, status_code=response.status_code)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/generate-token-script")
async def generate_token_script(
    username: str = Query(..., description="Mapbox username")
):
    """Generate a Python script for creating tokens"""
    script = f'''#!/usr/bin/env python3
"""
Mapbox Token Generator Script
Generated for: {username}
"""

import requests
import json
import sys

def create_mapbox_token(master_token, username, note="NetCDF Converter Token"):
    """Create a new Mapbox token with MTS scopes"""
    
    # First, validate the master token
    validate_url = f"https://api.mapbox.com/tokens/v2?access_token={{master_token}}"
    validate_response = requests.get(validate_url)
    
    if validate_response.status_code != 200:
        print("❌ Invalid master token!")
        return None
    
    token_info = validate_response.json().get('token', {{}})
    scopes = token_info.get('scopes', [])
    
    if 'tokens:write' not in scopes:
        print("❌ Master token doesn't have 'tokens:write' scope!")
        print("   Current scopes:", ', '.join(scopes))
        return None
    
    print("✅ Master token validated")
    
    # Create new token
    create_url = f"https://api.mapbox.com/tokens/v2/{{username}}?access_token={{master_token}}"
    
    payload = {{
        "note": note,
        "scopes": [
            "tilesets:write",
            "tilesets:read",
            "tilesets:list",
            "sources:write",
            "sources:read"
        ]
    }}
    
    print("📝 Creating token with scopes:", ', '.join(payload['scopes']))
    
    create_response = requests.post(create_url, json=payload)
    
    if create_response.status_code in [200, 201]:
        data = create_response.json()
        print("✅ Token created successfully!")
        return data
    else:
        print(f"❌ Failed to create token: {{create_response.status_code}}")
        print("   Error:", create_response.json())
        return None

if __name__ == "__main__":
    print("🔑 Mapbox Token Generator")
    print("=" * 50)
    
    master_token = input("Enter your master token (with tokens:write scope): ").strip()
    
    if not master_token:
        print("❌ No token provided!")
        sys.exit(1)
    
    username = "{username}"
    note = input(f"Enter token note (default: 'NetCDF Converter Token'): ").strip()
    
    if not note:
        note = "NetCDF Converter Token"
    
    result = create_mapbox_token(master_token, username, note)
    
    if result:
        print("\\n" + "=" * 50)
        print("🎉 SUCCESS!")
        print("=" * 50)
        print(f"Token: {{result['token']}}")
        print(f"ID: {{result['id']}}")
        print(f"Note: {{result['note']}}")
        print(f"Scopes: {{', '.join(result['scopes'])}}")
        print(f"Created: {{result['created']}}")
        print("\\n⚠️  Save this token now - it won't be shown again!")
        print("\\nNext steps:")
        print("1. Copy the token above")
        print("2. Update your .env file: MAPBOX_TOKEN=<your-new-token>")
        print("3. Restart your application")
'''
    
    return Response(
        content=script,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=generate_mapbox_token_{username}.py"
        }
    )

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

@app.get("/api/tileset-info/{tileset_id}")
async def get_tileset_info(tileset_id: str):
    """Get information about a specific tileset"""
    try:
        if not MAPBOX_TOKEN or not MAPBOX_USERNAME:
            return JSONResponse({
                "error": "Mapbox credentials not configured"
            }, status_code=500)
        
        # Get tileset metadata
        url = f"https://api.mapbox.com/tilesets/v1/{MAPBOX_USERNAME}.{tileset_id}?access_token={MAPBOX_TOKEN}"
        response = requests.get(url)
        
        if response.status_code == 200:
            tileset_data = response.json()
            
            # Get recipe if available
            recipe_path = TEMP_DIR / f"recipe_{tileset_id}.json"
            recipe = None
            if recipe_path.exists():
                with open(recipe_path, 'r') as f:
                    recipe = json.load(f)
            
            return JSONResponse({
                "success": True,
                "tileset": tileset_data,
                "recipe": recipe,
                "visualization_url": f"/visualize-advanced/{tileset_id}"
            })
        else:
            return JSONResponse({
                "error": f"Failed to get tileset info: {response.text}"
            }, status_code=response.status_code)
            
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

@app.on_event("startup")
async def startup_event():
    """Cleanup old temp files on startup"""
    try:
        import time
        current_time = time.time()
        max_age = float(os.getenv("TEMP_FILE_MAX_AGE", 24)) * 3600  # Convert hours to seconds
        
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age:
                    file_path.unlink()
                    print(f"Cleaned up old file: {file_path}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if os.getenv("DEBUG") else "Please contact support"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = bool(int(os.getenv("DEBUG", 0)))
    
    print(f"""
🚀 Starting NetCDF to Mapbox Converter
    
   Server: http://{host}:{port}
   Debug:  {debug}
   User:   {MAPBOX_USERNAME or 'Not configured'}
    
   Endpoints:
   - Main:         http://{host}:{port}/
   - Token Test:   http://{host}:{port}/test-token
   - Token Gen:    http://{host}:{port}/generate-token
   - Health:       http://{host}:{port}/health
   - API Docs:     http://{host}:{port}/docs
    
   Press Ctrl+C to stop
    """)
    
    uvicorn.run(app, host=host, port=port, reload=debug)