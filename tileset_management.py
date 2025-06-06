import requests
import json
import time
from typing import Dict, List, Optional
import boto3
from pathlib import Path
import xarray as xr

class MapboxTilesetManager:
    """Manage Mapbox tilesets with MTS API"""
    
    def __init__(self, token: str, username: str):
        self.token = token
        self.username = username
        self.base_url = "https://api.mapbox.com"
        
    def create_tileset_source(self, source_id: str, file_path: str) -> Dict:
        """Upload a file to create a tileset source"""
        # Get S3 credentials
        cred_url = f"{self.base_url}/tilesets/v1/sources/{self.username}/{source_id}/upload-credentials?access_token={self.token}"
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
        
        with open(file_path, 'rb') as f:
            s3_client.put_object(
                Bucket=creds['bucket'],
                Key=creds['key'],
                Body=f
            )
        
        return {
            "source_id": f"{self.username}/{source_id}",
            "status": "uploaded"
        }
    
    def create_tileset(self, tileset_id: str, recipe: Dict, name: str = None, description: str = None) -> Dict:
        """Create a new tileset with recipe"""
        tileset_url = f"{self.base_url}/tilesets/v1/{self.username}.{tileset_id}?access_token={self.token}"
        
        data = {
            "recipe": recipe,
            "name": name or f"NetCDF Tileset - {tileset_id}",
            "description": description or "Multi-variable weather data tileset"
        }
        
        create_resp = requests.post(tileset_url, json=data)
        
        if create_resp.status_code == 409:  # Already exists
            # Update the recipe
            update_resp = requests.patch(tileset_url, json={"recipe": recipe})
            if update_resp.status_code != 200:
                raise Exception(f"Failed to update tileset: {update_resp.text}")
            return {"tileset_id": f"{self.username}.{tileset_id}", "status": "updated"}
        
        if create_resp.status_code not in [200, 201]:
            raise Exception(f"Failed to create tileset: {create_resp.text}")
        
        return {"tileset_id": f"{self.username}.{tileset_id}", "status": "created"}
    
    def publish_tileset(self, tileset_id: str) -> Dict:
        """Publish a tileset to make it available"""
        publish_url = f"{self.base_url}/tilesets/v1/{self.username}.{tileset_id}/publish?access_token={self.token}"
        publish_resp = requests.post(publish_url)
        
        if publish_resp.status_code != 200:
            raise Exception(f"Failed to publish tileset: {publish_resp.text}")
        
        return publish_resp.json()
    
    def get_tileset_status(self, tileset_id: str) -> Dict:
        """Get the current status of a tileset"""
        status_url = f"{self.base_url}/tilesets/v1/{self.username}.{tileset_id}?access_token={self.token}"
        resp = requests.get(status_url)
        
        if resp.status_code != 200:
            raise Exception(f"Failed to get tileset status: {resp.text}")
        
        return resp.json()
    
    def get_job_status(self, job_id: str) -> Dict:
        """Check the status of a publish job"""
        job_url = f"{self.base_url}/tilesets/v1/{self.username}/jobs/{job_id}?access_token={self.token}"
        resp = requests.get(job_url)
        
        if resp.status_code != 200:
            raise Exception(f"Failed to get job status: {resp.text}")
        
        return resp.json()
    
    def wait_for_job(self, job_id: str, timeout: int = 300) -> Dict:
        """Wait for a job to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            if status.get('stage') == 'success':
                return {"status": "success", "job": status}
            elif status.get('stage') == 'failed':
                return {"status": "failed", "job": status, "error": status.get('errors')}
            
            time.sleep(5)  # Check every 5 seconds
        
        return {"status": "timeout", "job_id": job_id}
    
    def query_tileset_statistics(self, tileset_id: str) -> Dict:
        """Get statistics about a tileset"""
        stats_url = f"{self.base_url}/tilesets/v1/{self.username}.{tileset_id}/statistics?access_token={self.token}"
        resp = requests.get(stats_url)
        
        if resp.status_code != 200:
            raise Exception(f"Failed to get tileset statistics: {resp.text}")
        
        return resp.json()
    
    def list_tilesets(self, limit: int = 100) -> List[Dict]:
        """List all tilesets for the user"""
        list_url = f"{self.base_url}/tilesets/v1/{self.username}?access_token={self.token}&limit={limit}"
        resp = requests.get(list_url)
        
        if resp.status_code != 200:
            raise Exception(f"Failed to list tilesets: {resp.text}")
        
        return resp.json()
    
    def delete_tileset(self, tileset_id: str) -> bool:
        """Delete a tileset"""
        delete_url = f"{self.base_url}/tilesets/v1/{self.username}.{tileset_id}?access_token={self.token}"
        resp = requests.delete(delete_url)
        
        return resp.status_code == 204
    
    def update_tileset_source(self, source_id: str, file_path: str) -> Dict:
        """Update an existing tileset source with new data"""
        # First delete the old source if it exists
        delete_url = f"{self.base_url}/tilesets/v1/sources/{self.username}/{source_id}?access_token={self.token}"
        requests.delete(delete_url)  # Ignore if it doesn't exist
        
        # Create new source
        return self.create_tileset_source(source_id, file_path)
    
    def process_netcdf_to_tileset(self, nc_path: str, tileset_id: str, recipe: Dict = None) -> Dict:
        """
        Complete workflow to process NetCDF to Mapbox tileset
        """
        try:
            # 1. Create source
            source_id = f"{tileset_id}_source"
            print(f"Creating tileset source: {source_id}")
            source_result = self.create_tileset_source(source_id, nc_path)
            
            # 2. Create or update tileset with recipe
            if not recipe:
                from app import create_recipe_for_netcdf
                recipe = create_recipe_for_netcdf(nc_path, tileset_id, self.username)
            
            print(f"Creating tileset: {tileset_id}")
            tileset_result = self.create_tileset(tileset_id, recipe)
            
            # 3. Publish tileset
            print(f"Publishing tileset: {tileset_id}")
            publish_result = self.publish_tileset(tileset_id)
            job_id = publish_result.get('jobId')
            
            # 4. Wait for completion
            if job_id:
                print(f"Waiting for job {job_id} to complete...")
                job_result = self.wait_for_job(job_id)
                
                if job_result['status'] == 'success':
                    return {
                        "success": True,
                        "tileset_id": f"{self.username}.{tileset_id}",
                        "source_id": source_result['source_id'],
                        "job_id": job_id,
                        "recipe": recipe,
                        "status": "completed"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Job failed: {job_result}",
                        "tileset_id": f"{self.username}.{tileset_id}",
                        "job_id": job_id
                    }
            
            return {
                "success": True,
                "tileset_id": f"{self.username}.{tileset_id}",
                "source_id": source_result['source_id'],
                "recipe": recipe,
                "status": "published"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Utility functions for queries
class TilesetQueryTools:
    """Tools for querying tileset data"""
    
    @staticmethod
    def query_temporal_range(ds: xr.Dataset, start_time: str = None, end_time: str = None) -> xr.Dataset:
        """Query data within a temporal range"""
        if 'time' in ds.dims:
            if start_time and end_time:
                return ds.sel(time=slice(start_time, end_time))
            elif start_time:
                return ds.sel(time=ds.time >= start_time)
            elif end_time:
                return ds.sel(time=ds.time <= end_time)
        return ds
    
    @staticmethod
    def query_spatial_bounds(ds: xr.Dataset, west: float, east: float, south: float, north: float) -> xr.Dataset:
        """Query data within spatial bounds"""
        if 'longitude' in ds.dims and 'latitude' in ds.dims:
            return ds.sel(
                longitude=slice(west, east),
                latitude=slice(south, north)
            )
        elif 'lon' in ds.dims and 'lat' in ds.dims:
            return ds.sel(
                lon=slice(west, east),
                lat=slice(south, north)
            )
        return ds
    
    @staticmethod
    def compute_statistics(ds: xr.Dataset, variables: List[str] = None) -> Dict:
        """Compute statistics for variables"""
        stats = {}
        
        if variables is None:
            variables = list(ds.data_vars)
        
        for var in variables:
            if var in ds:
                stats[var] = {
                    "min": float(ds[var].min().values),
                    "max": float(ds[var].max().values),
                    "mean": float(ds[var].mean().values),
                    "std": float(ds[var].std().values),
                    "count": int(ds[var].count().values)
                }
        
        return stats