import xarray as xr
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import json
from datetime import datetime, timedelta

class EnhancedTilesetQueryTools:
    """Enhanced tools for querying multi-variable tileset data"""
    
    @staticmethod
    def extract_wind_vectors_from_tileset(tileset_id: str, bounds: Tuple[float, float, float, float], 
                                        zoom: int, token: str) -> Dict:
        """
        Extract wind vector data from a Mapbox tileset for particle animation
        
        Args:
            tileset_id: Mapbox tileset ID
            bounds: (west, south, east, north) bounds
            zoom: Zoom level for tile queries
            token: Mapbox access token
            
        Returns:
            Dictionary with wind field data
        """
        west, south, east, north = bounds
        
        # Calculate tile coordinates for the bounds
        tiles = EnhancedTilesetQueryTools._get_tiles_for_bounds(bounds, zoom)
        
        wind_data = {
            'u_component': [],
            'v_component': [],
            'grid': {
                'width': 0,
                'height': 0,
                'bounds': bounds,
                'resolution': 0
            }
        }
        
        # In production, fetch actual tiles and extract wind data
        # For now, return structure
        return wind_data
    
    @staticmethod
    def query_multi_variable_statistics(ds: xr.Dataset, 
                                      variables: List[str] = None,
                                      region: Dict = None,
                                      time_range: Tuple[str, str] = None) -> Dict:
        """
        Compute comprehensive statistics for multiple variables with optional filtering
        
        Args:
            ds: xarray Dataset
            variables: List of variables to analyze (None for all)
            region: Dict with 'west', 'east', 'south', 'north' bounds
            time_range: Tuple of (start_time, end_time) in ISO format
            
        Returns:
            Dictionary with statistics for each variable
        """
        # Apply spatial filtering
        if region:
            ds = EnhancedTilesetQueryTools._apply_spatial_filter(ds, region)
        
        # Apply temporal filtering
        if time_range:
            ds = EnhancedTilesetQueryTools._apply_temporal_filter(ds, time_range)
        
        if variables is None:
            variables = list(ds.data_vars)
        
        stats = {}
        
        for var in variables:
            if var not in ds.data_vars:
                continue
                
            data = ds[var]
            
            # Handle time dimension
            if 'time' in data.dims and len(data.time) > 0:
                data = data.isel(time=0)
            
            # Basic statistics
            try:
                var_stats = {
                    'min': float(data.min().values),
                    'max': float(data.max().values),
                    'mean': float(data.mean().values),
                    'std': float(data.std().values) if data.size > 1 else 0,
                    'median': float(data.median().values) if data.size > 1 else float(data.mean().values),
                    'count': int(data.count().values),
                    'units': data.attrs.get('units', 'unknown'),
                    'long_name': data.attrs.get('long_name', var)
                }
                
                # Percentiles
                if data.size > 10:
                    percentiles = [5, 25, 75, 95]
                    for p in percentiles:
                        var_stats[f'p{p}'] = float(data.quantile(p/100).values)
                
                # Spatial distribution
                if 'latitude' in data.dims and 'longitude' in data.dims:
                    var_stats['spatial'] = {
                        'lat_range': [float(data.latitude.min()), float(data.latitude.max())],
                        'lon_range': [float(data.longitude.min()), float(data.longitude.max())],
                        'grid_points': data.latitude.size * data.longitude.size
                    }
                
                # Temporal distribution
                if 'time' in ds[var].dims:
                    var_stats['temporal'] = {
                        'start': str(ds[var].time.min().values),
                        'end': str(ds[var].time.max().values),
                        'steps': ds[var].time.size,
                        'frequency': EnhancedTilesetQueryTools._infer_temporal_frequency(ds[var].time)
                    }
                
                stats[var] = var_stats
                
            except Exception as e:
                print(f"Error computing statistics for {var}: {e}")
                stats[var] = {'error': str(e)}
        
        # Add vector field analysis if u/v components exist
        vector_stats = EnhancedTilesetQueryTools._analyze_vector_fields(ds, variables)
        if vector_stats:
            stats['vector_fields'] = vector_stats
        
        return stats
    
    @staticmethod
    def create_wind_grid_for_visualization(ds: xr.Dataset, 
                                          u_var: str, v_var: str,
                                          target_resolution: int = 50) -> Dict:
        """
        Create a wind grid optimized for particle visualization
        
        Args:
            ds: xarray Dataset
            u_var: Name of u-component variable
            v_var: Name of v-component variable  
            target_resolution: Target grid resolution (points per dimension)
            
        Returns:
            Dictionary with wind grid data
        """
        if u_var not in ds.data_vars or v_var not in ds.data_vars:
            raise ValueError(f"Variables {u_var} or {v_var} not found in dataset")
        
        u_data = ds[u_var]
        v_data = ds[v_var]
        
        # Handle time dimension
        if 'time' in u_data.dims and len(u_data.time) > 0:
            u_data = u_data.isel(time=0)
            v_data = v_data.isel(time=0)
        
        # Get coordinate names
        lat_name = 'latitude' if 'latitude' in u_data.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in u_data.dims else 'lon'
        
        # Resample to target resolution if needed
        current_res = max(u_data[lat_name].size, u_data[lon_name].size)
        if current_res > target_resolution * 2:
            # Downsample for performance
            lat_step = max(1, current_res // target_resolution)
            lon_step = max(1, current_res // target_resolution)
            u_data = u_data.isel({lat_name: slice(0, None, lat_step), 
                                 lon_name: slice(0, None, lon_step)})
            v_data = v_data.isel({lat_name: slice(0, None, lat_step), 
                                 lon_name: slice(0, None, lon_step)})
        
        # Create meshgrid
        lats = u_data[lat_name].values
        lons = u_data[lon_name].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Calculate wind statistics
        u_values = u_data.values
        v_values = v_data.values
        speed = np.sqrt(u_values**2 + v_values**2)
        direction = np.arctan2(v_values, u_values) * 180 / np.pi
        
        return {
            'grid': {
                'lats': lats.tolist(),
                'lons': lons.tolist(),
                'lat_grid': lat_grid.tolist(),
                'lon_grid': lon_grid.tolist(),
                'shape': list(u_values.shape)
            },
            'u_component': u_values.tolist(),
            'v_component': v_values.tolist(),
            'speed': {
                'values': speed.tolist(),
                'min': float(np.nanmin(speed)),
                'max': float(np.nanmax(speed)),
                'mean': float(np.nanmean(speed))
            },
            'direction': direction.tolist(),
            'bounds': {
                'west': float(lons.min()),
                'east': float(lons.max()),
                'south': float(lats.min()),
                'north': float(lats.max())
            },
            'metadata': {
                'u_variable': u_var,
                'v_variable': v_var,
                'units': u_data.attrs.get('units', 'm/s'),
                'generated_at': datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def analyze_temporal_patterns(ds: xr.Dataset, variable: str, 
                                frequency: str = 'D') -> Dict:
        """
        Analyze temporal patterns in the data
        
        Args:
            ds: xarray Dataset
            variable: Variable to analyze
            frequency: Temporal frequency for aggregation ('H', 'D', 'M', etc.)
            
        Returns:
            Dictionary with temporal analysis results
        """
        if variable not in ds.data_vars:
            raise ValueError(f"Variable {variable} not found in dataset")
        
        data = ds[variable]
        
        if 'time' not in data.dims:
            return {'error': 'No time dimension found'}
        
        # Resample to specified frequency
        resampled = data.resample(time=frequency)
        
        patterns = {
            'frequency': frequency,
            'original_timesteps': data.time.size,
            'time_range': {
                'start': str(data.time.min().values),
                'end': str(data.time.max().values)
            }
        }
        
        try:
            # Compute aggregated statistics
            patterns['aggregations'] = {
                'mean': resampled.mean().values.tolist(),
                'max': resampled.max().values.tolist(),
                'min': resampled.min().values.tolist(),
                'std': resampled.std().values.tolist()
            }
            patterns['aggregated_timesteps'] = len(patterns['aggregations']['mean'])
            
            # Detect trends
            if data.time.size > 10:
                # Get spatial mean time series
                spatial_dims = [d for d in data.dims if d != 'time']
                if spatial_dims:
                    spatial_mean = data.mean(dim=spatial_dims)
                else:
                    spatial_mean = data
                
                # Convert time to numeric for trend analysis
                time_numeric = np.arange(len(spatial_mean.time))
                
                # Simple linear regression for trend
                coeffs = np.polyfit(time_numeric, spatial_mean.values, 1)
                patterns['trend'] = {
                    'slope': float(coeffs[0]),
                    'intercept': float(coeffs[1]),
                    'direction': 'increasing' if coeffs[0] > 0 else 'decreasing'
                }
            
            # Detect diurnal cycle (if hourly data)
            if data.time.size > 24 and hasattr(data.time, 'dt'):
                try:
                    hourly_mean = data.groupby(data.time.dt.hour).mean()
                    if hourly_mean.size == 24:
                        patterns['diurnal_cycle'] = {
                            'hourly_means': hourly_mean.values.tolist(),
                            'peak_hour': int(hourly_mean.argmax().values),
                            'minimum_hour': int(hourly_mean.argmin().values),
                            'amplitude': float(hourly_mean.max() - hourly_mean.min())
                        }
                except:
                    pass
                    
        except Exception as e:
            patterns['error'] = f"Error in temporal analysis: {str(e)}"
        
        return patterns
    
    @staticmethod
    def extract_point_time_series(ds: xr.Dataset, variable: str, 
                                 lat: float, lon: float) -> Dict:
        """
        Extract time series at a specific point
        
        Args:
            ds: xarray Dataset
            variable: Variable name
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with time series data
        """
        if variable not in ds.data_vars:
            raise ValueError(f"Variable {variable} not found")
        
        data = ds[variable]
        
        # Get coordinate names
        lat_name = 'latitude' if 'latitude' in data.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in data.dims else 'lon'
        
        # Find nearest point
        try:
            point_data = data.sel({lat_name: lat, lon_name: lon}, method='nearest')
            
            result = {
                'location': {
                    'requested': {'lat': lat, 'lon': lon},
                    'actual': {
                        'lat': float(point_data[lat_name].values),
                        'lon': float(point_data[lon_name].values)
                    }
                },
                'variable': variable,
                'units': data.attrs.get('units', 'unknown')
            }
            
            if 'time' in point_data.dims:
                result['time_series'] = {
                    'times': [str(t) for t in point_data.time.values],
                    'values': point_data.values.tolist(),
                    'statistics': {
                        'min': float(point_data.min()),
                        'max': float(point_data.max()),
                        'mean': float(point_data.mean()),
                        'std': float(point_data.std()) if point_data.size > 1 else 0
                    }
                }
            else:
                result['value'] = float(point_data.values)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    # Helper methods
    @staticmethod
    def _apply_spatial_filter(ds: xr.Dataset, region: Dict) -> xr.Dataset:
        """Apply spatial filtering to dataset"""
        lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
        
        return ds.sel({
            lat_name: slice(region['south'], region['north']),
            lon_name: slice(region['west'], region['east'])
        })
    
    @staticmethod
    def _apply_temporal_filter(ds: xr.Dataset, time_range: Tuple[str, str]) -> xr.Dataset:
        """Apply temporal filtering to dataset"""
        if 'time' in ds.dims:
            return ds.sel(time=slice(time_range[0], time_range[1]))
        return ds
    
    @staticmethod
    def _infer_temporal_frequency(time_coord) -> str:
        """Infer temporal frequency from time coordinate"""
        if len(time_coord) < 2:
            return 'unknown'
        
        try:
            time_diff = time_coord[1] - time_coord[0]
            
            # Convert to timedelta64
            if hasattr(time_diff, 'values'):
                time_diff = time_diff.values
            
            # Convert to hours
            hours = time_diff / np.timedelta64(1, 'h')
            
            if hours <= 1:
                return 'hourly'
            elif hours <= 24:
                return f'{int(hours)}-hourly'
            elif hours <= 24 * 7:
                return 'daily' if hours == 24 else f'{int(hours/24)}-daily'
            elif hours <= 24 * 31:
                return 'weekly' if hours == 24 * 7 else f'{int(hours/(24*7))}-weekly'
            else:
                return 'monthly'
                
        except:
            return 'unknown'
    
    @staticmethod
    def _analyze_vector_fields(ds: xr.Dataset, variables: List[str]) -> Dict:
        """Analyze vector fields in the dataset"""
        vector_patterns = [
            ('u', 'v'), ('u10', 'v10'), ('water_u', 'water_v'),
            ('eastward_', 'northward_'), ('_u', '_v')
        ]
        
        vector_fields = {}
        
        for u_pattern, v_pattern in vector_patterns:
            u_vars = [v for v in variables if u_pattern in v.lower()]
            v_vars = [v for v in variables if v_pattern in v.lower()]
            
            if u_vars and v_vars:
                # Try to match pairs
                for u_var in u_vars:
                    v_var = u_var.replace(u_pattern, v_pattern)
                    if v_var in v_vars and u_var in ds.data_vars and v_var in ds.data_vars:
                        u_data = ds[u_var]
                        v_data = ds[v_var]
                        
                        # Handle time dimension
                        if 'time' in u_data.dims and len(u_data.time) > 0:
                            u_data = u_data.isel(time=0)
                            v_data = v_data.isel(time=0)
                        
                        # Calculate magnitude and direction
                        magnitude = np.sqrt(u_data**2 + v_data**2)
                        direction = np.arctan2(v_data, u_data) * 180 / np.pi
                        
                        field_name = 'wind' if 'wind' in u_var.lower() else 'flow'
                        
                        vector_fields[field_name] = {
                            'u_component': u_var,
                            'v_component': v_var,
                            'magnitude': {
                                'min': float(magnitude.min().values),
                                'max': float(magnitude.max().values),
                                'mean': float(magnitude.mean().values),
                                'std': float(magnitude.std().values) if magnitude.size > 1 else 0
                            },
                            'direction': {
                                'dominant': float(np.nanmean(direction)),
                                'variability': float(np.nanstd(direction)) if magnitude.size > 1 else 0
                            }
                        }
                        break
        
        return vector_fields
    
    @staticmethod
    def _get_tiles_for_bounds(bounds: Tuple[float, float, float, float], zoom: int) -> List[Dict]:
        """Calculate tile coordinates for given bounds and zoom level"""
        west, south, east, north = bounds
        
        # Convert lat/lon to tile coordinates
        def lat_lon_to_tile(lat, lon, zoom):
            lat_rad = np.radians(lat)
            n = 2.0 ** zoom
            x = int((lon + 180.0) / 360.0 * n)
            y = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
            return x, y
        
        min_x, max_y = lat_lon_to_tile(south, west, zoom)
        max_x, min_y = lat_lon_to_tile(north, east, zoom)
        
        tiles = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tiles.append({'x': x, 'y': y, 'z': zoom})
        
        return tiles