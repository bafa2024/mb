import xarray as xr
import numpy as np
import re
import traceback
from typing import Dict, List, Tuple
from datetime import datetime

def create_enhanced_recipe_for_netcdf(nc_path: str, tileset_id: str, username: str) -> Dict:
    """
    Create an enhanced Mapbox recipe for NetCDF data with proper RasterArray format
    for both vector and scalar fields, following the Mapbox wind example pattern.
    
    Args:
        nc_path: Path to NetCDF file
        tileset_id: Tileset ID for Mapbox
        username: Mapbox username
        
    Returns:
        Dict: Mapbox MTS recipe with RasterArray configuration
    """
    try:
        ds = xr.open_dataset(nc_path)
        
        # Initialize recipe structure
        recipe = {
            "version": 1,
            "layers": {}
        }
        
        # Identify variables and their types
        vector_pairs = []
        scalar_vars = []
        all_vars = list(ds.data_vars)
        processed_vars = set()
        
        # Common vector component patterns
        vector_patterns = [
            ('u10', 'v10'),  # 10m wind components
            ('u', 'v'),      # Generic u/v components
            ('water_u', 'water_v'),  # Ocean currents
            ('eastward_wind', 'northward_wind'),
            ('u_wind', 'v_wind'),
            ('uwind', 'vwind'),
            ('uwnd', 'vwnd'),
            ('u_component', 'v_component')
        ]
        
        # Find vector pairs with more flexible matching
        for u_pattern, v_pattern in vector_patterns:
            u_matches = []
            v_matches = []
            
            for var in all_vars:
                var_lower = var.lower()
                if u_pattern in var_lower and var not in processed_vars:
                    u_matches.append(var)
                if v_pattern in var_lower and var not in processed_vars:
                    v_matches.append(var)
            
            if u_matches and v_matches:
                # Try to match pairs by common prefix/suffix
                for u_var in u_matches:
                    for v_var in v_matches:
                        # Check if they share a common base
                        u_base = u_var.replace(u_pattern, '')
                        v_base = v_var.replace(v_pattern, '')
                        if u_base == v_base or (u_var.replace('u', 'v') == v_var):
                            vector_pairs.append({
                                'u': u_var,
                                'v': v_var,
                                'name': '10winds' if '10' in u_var else 'winds'
                            })
                            processed_vars.add(u_var)
                            processed_vars.add(v_var)
                            break
        
        # All other variables are scalar
        scalar_vars = [v for v in all_vars if v not in processed_vars]
        
        # Create main raster layer with all variables as bands
        band_index = 0
        bands_config = {}
        
        # Add vector components first (following Mapbox wind example pattern)
        for vector_pair in vector_pairs:
            # U component
            band_index += 1
            u_band_name = f"{vector_pair['name']}_u"
            bands_config[u_band_name] = {
                "band": band_index,
                "source_band": vector_pair['u'],
                "bidx": band_index
            }
            
            # V component
            band_index += 1
            v_band_name = f"{vector_pair['name']}_v"
            bands_config[v_band_name] = {
                "band": band_index,
                "source_band": vector_pair['v'],
                "bidx": band_index
            }
        
        # Add scalar variables
        for scalar_var in scalar_vars:
            band_index += 1
            # Clean variable name for band name
            band_name = re.sub(r'[^a-zA-Z0-9_]', '_', scalar_var.lower())
            bands_config[band_name] = {
                "band": band_index,
                "source_band": scalar_var,
                "bidx": band_index
            }
        
        # Create the main layer configuration
        recipe["layers"]["default"] = {
            "source": f"mapbox://tileset-source/{username}/{tileset_id}",
            "minzoom": 0,
            "maxzoom": 14,
            "tilesets": {
                f"{username}.{tileset_id}": {
                    "type": "raster",
                    "buffer_size": 64,
                    "encoding": "terrarium",
                    "resolution": 512,
                    "tiles": {
                        "resampling": "bilinear",
                        "bands": list(range(1, band_index + 1))
                    },
                    "raster_array": {
                        "bands": bands_config
                    }
                }
            }
        }
        
        # Add named layers for specific variables (for easier GL JS access)
        # Wind layer (if vector pairs exist)
        if vector_pairs:
            wind_pair = vector_pairs[0]  # Use first vector pair
            wind_bands = {
                f"{wind_pair['name']}_u": bands_config[f"{wind_pair['name']}_u"],
                f"{wind_pair['name']}_v": bands_config[f"{wind_pair['name']}_v"]
            }
            
            recipe["layers"][wind_pair['name']] = {
                "source": f"mapbox://tileset-source/{username}/{tileset_id}",
                "minzoom": 0,
                "maxzoom": 14,
                "tilesets": {
                    f"{username}.{tileset_id}": {
                        "type": "raster",
                        "buffer_size": 64,
                        "encoding": "terrarium",
                        "resolution": 512,
                        "tiles": {
                            "resampling": "bilinear",
                            "bands": [wind_bands[f"{wind_pair['name']}_u"]["band"], 
                                     wind_bands[f"{wind_pair['name']}_v"]["band"]]
                        },
                        "raster_array": {
                            "bands": wind_bands
                        }
                    }
                }
            }
        
        # Individual scalar layers for primary variables
        primary_scalars = []
        for scalar_var in scalar_vars[:3]:  # Limit to first 3 scalar variables
            band_name = re.sub(r'[^a-zA-Z0-9_]', '_', scalar_var.lower())
            if band_name in bands_config:
                primary_scalars.append(band_name)
                
                recipe["layers"][band_name] = {
                    "source": f"mapbox://tileset-source/{username}/{tileset_id}",
                    "minzoom": 0,
                    "maxzoom": 14,
                    "tilesets": {
                        f"{username}.{tileset_id}": {
                            "type": "raster",
                            "buffer_size": 64,
                            "encoding": "terrarium",
                            "resolution": 512,
                            "tiles": {
                                "resampling": "bilinear",
                                "bands": [bands_config[band_name]["band"]]
                            },
                            "raster_array": {
                                "bands": {
                                    band_name: bands_config[band_name]
                                }
                            }
                        }
                    }
                }
        
        # Add metadata for reference
        recipe["metadata"] = {
            "band_mapping": {v['source_band']: k for k, v in bands_config.items()},
            "vector_pairs": vector_pairs,
            "scalar_variables": scalar_vars,
            "primary_scalars": primary_scalars,
            "total_bands": band_index,
            "bands_info": {},
            "created_at": datetime.now().isoformat(),
            "source_file": nc_path.split('/')[-1]
        }
        
        # Add detailed band information with statistics
        for band_name, band_config in bands_config.items():
            source_var = band_config['source_band']
            if source_var in ds.data_vars:
                data = ds[source_var]
                
                # Handle time dimension
                if 'time' in data.dims:
                    data = data.isel(time=0)
                
                # Convert to float32 for safe computation
                data_values = data.values.astype(np.float32)
                
                # Compute statistics safely
                valid_data = data_values[~np.isnan(data_values)]
                if len(valid_data) > 0:
                    stats = {
                        "min": float(np.min(valid_data)),
                        "max": float(np.max(valid_data)),
                        "mean": float(np.mean(valid_data)),
                        "std": float(np.std(valid_data)) if len(valid_data) > 1 else 0
                    }
                else:
                    stats = {
                        "min": 0,
                        "max": 1,
                        "mean": 0.5,
                        "std": 0
                    }
                
                recipe["metadata"]["bands_info"][band_name] = {
                    "source_variable": source_var,
                    "band_index": band_config['band'],
                    "type": "vector_component" if any(band_name.endswith(suffix) for suffix in ['_u', '_v']) else "scalar",
                    "stats": stats,
                    "units": data.attrs.get('units', 'unknown'),
                    "long_name": data.attrs.get('long_name', source_var),
                    "dimensions": list(data.dims),
                    "shape": list(data.shape)
                }
        
        # Add dataset-level metadata
        recipe["metadata"]["dataset_info"] = {
            "dimensions": dict(ds.dims),
            "coordinates": list(ds.coords),
            "attributes": dict(ds.attrs),
            "time_coverage": None
        }
        
        # Check for time coverage
        if 'time' in ds.coords:
            time_coord = ds.coords['time']
            recipe["metadata"]["dataset_info"]["time_coverage"] = {
                "start": str(time_coord.min().values),
                "end": str(time_coord.max().values),
                "steps": len(time_coord)
            }
        
        ds.close()
        return recipe
        
    except Exception as e:
        print(f"Error creating enhanced recipe: {str(e)}")
        traceback.print_exc()
        raise

def validate_recipe(recipe: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a Mapbox MTS recipe
    
    Args:
        recipe: Recipe dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    if "version" not in recipe:
        errors.append("Missing 'version' field")
    
    if "layers" not in recipe:
        errors.append("Missing 'layers' field")
    
    # Check layer structure
    for layer_name, layer_config in recipe.get("layers", {}).items():
        if "source" not in layer_config:
            errors.append(f"Layer '{layer_name}' missing 'source' field")
        
        if "tilesets" not in layer_config:
            errors.append(f"Layer '{layer_name}' missing 'tilesets' field")
        
        # Check tileset configuration
        for tileset_id, tileset_config in layer_config.get("tilesets", {}).items():
            if "raster_array" not in tileset_config:
                errors.append(f"Tileset '{tileset_id}' missing 'raster_array' field")
            
            # Check bands
            bands = tileset_config.get("raster_array", {}).get("bands", {})
            if not bands:
                errors.append(f"Tileset '{tileset_id}' has no bands defined")
            
            for band_name, band_config in bands.items():
                if "band" not in band_config:
                    errors.append(f"Band '{band_name}' missing 'band' index")
                if "source_band" not in band_config:
                    errors.append(f"Band '{band_name}' missing 'source_band'")
    
    return len(errors) == 0, errors

def optimize_recipe_for_visualization(recipe: Dict) -> Dict:
    """
    Optimize recipe for better visualization performance
    
    Args:
        recipe: Original recipe
        
    Returns:
        Optimized recipe
    """
    optimized = recipe.copy()
    
    # Adjust tile settings for better performance
    for layer_config in optimized.get("layers", {}).values():
        for tileset_config in layer_config.get("tilesets", {}).values():
            # Optimize buffer size based on use case
            if "buffer_size" in tileset_config:
                # Larger buffer for smoother edges
                tileset_config["buffer_size"] = 128
            
            # Optimize resolution
            if "resolution" in tileset_config:
                # Higher resolution for detailed data
                tileset_config["resolution"] = 512
            
            # Ensure bilinear resampling for smooth visualization
            if "tiles" in tileset_config:
                tileset_config["tiles"]["resampling"] = "bilinear"
    
    return optimized