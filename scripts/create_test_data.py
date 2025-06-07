#!/usr/bin/env python3
"""
Create test NetCDF files with time dimensions for testing animations
"""

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_weather_test_data(output_file='test_weather_animated.nc'):
    """
    Create a test NetCDF file with weather data including time dimension
    """
    # Define dimensions
    lat = np.linspace(-90, 90, 90)  # 2-degree resolution
    lon = np.linspace(-180, 180, 180)  # 2-degree resolution
    time = pd.date_range('2024-01-01', periods=24, freq='H')  # 24 hours
    
    # Create coordinate arrays
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    
    # Initialize data arrays
    temp_data = np.zeros((len(time), len(lat), len(lon)))
    u_wind_data = np.zeros((len(time), len(lat), len(lon)))
    v_wind_data = np.zeros((len(time), len(lat), len(lon)))
    pressure_data = np.zeros((len(time), len(lat), len(lon)))
    
    # Generate data for each time step
    for t in range(len(time)):
        # Temperature: Diurnal cycle + latitude variation
        hour = t % 24
        temp_base = 15 + 25 * np.cos(np.radians(lat_2d))  # Latitude variation
        diurnal = 5 * np.sin(2 * np.pi * (hour - 6) / 24)  # Diurnal cycle
        temp_data[t] = temp_base + diurnal + np.random.randn(len(lat), len(lon)) * 2
        
        # Wind: Rotating pattern
        angle = 2 * np.pi * t / 24  # Full rotation in 24 hours
        wind_speed = 10 + 5 * np.sin(np.radians(lat_2d))
        u_wind_data[t] = wind_speed * np.cos(angle + np.radians(lon_2d) * 0.1)
        v_wind_data[t] = wind_speed * np.sin(angle + np.radians(lon_2d) * 0.1)
        
        # Pressure: Moving high pressure system
        center_lon = -180 + (360 * t / 24)  # Move across map
        distance = np.sqrt((lon_2d - center_lon)**2 + lat_2d**2)
        pressure_data[t] = 1013 + 20 * np.exp(-distance**2 / 1000) + np.random.randn(len(lat), len(lon))
    
    # Create dataset
    ds = xr.Dataset({
        'temperature': (['time', 'latitude', 'longitude'], temp_data, 
                       {'units': 'celsius', 'long_name': 'Air Temperature at 2m'}),
        'u10': (['time', 'latitude', 'longitude'], u_wind_data,
                {'units': 'm/s', 'long_name': 'U component of wind at 10m'}),
        'v10': (['time', 'latitude', 'longitude'], v_wind_data,
                {'units': 'm/s', 'long_name': 'V component of wind at 10m'}),
        'pressure': (['time', 'latitude', 'longitude'], pressure_data,
                    {'units': 'hPa', 'long_name': 'Sea Level Pressure'}),
    }, coords={
        'time': time,
        'latitude': lat,
        'longitude': lon,
    })
    
    # Add global attributes
    ds.attrs['title'] = 'Test Weather Data with Animation'
    ds.attrs['institution'] = 'Test Data Generator'
    ds.attrs['source'] = 'Synthetic data for testing'
    ds.attrs['history'] = f'Created on {datetime.now()}'
    ds.attrs['Conventions'] = 'CF-1.6'
    
    # Save to NetCDF
    ds.to_netcdf(output_file)
    print(f"Created {output_file}")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Time steps: {len(time)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return ds

def create_simple_test_data(output_file='test_simple.nc'):
    """
    Create a simple test NetCDF file without time dimension
    """
    # Define dimensions
    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(-180, 180, 360)
    
    # Create coordinate arrays
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    
    # Generate data
    temp = 15 + 10 * np.sin(np.radians(lat_2d)) * np.cos(np.radians(lon_2d))
    u_wind = 5 * np.cos(np.radians(lat_2d + lon_2d))
    v_wind = 5 * np.sin(np.radians(lat_2d + lon_2d))
    
    # Create dataset
    ds = xr.Dataset({
        'temperature': (['latitude', 'longitude'], temp),
        'u_wind': (['latitude', 'longitude'], u_wind),
        'v_wind': (['latitude', 'longitude'], v_wind),
    }, coords={
        'latitude': lat,
        'longitude': lon,
    })
    
    # Save to NetCDF
    ds.to_netcdf(output_file)
    print(f"Created {output_file}")
    
    return ds

if __name__ == "__main__":
    import os
    
    # Create test data files
    print("Creating test NetCDF files...")
    
    # Create animated data
    create_weather_test_data('test_weather_animated.nc')
    
    # Create simple data
    create_simple_test_data('test_simple.nc')
    
    print("\nTest files created successfully!")
    print("You can now upload these files to test the visualization features.")