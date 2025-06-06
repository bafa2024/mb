#!/usr/bin/env python3
"""
Command-line utilities for Mapbox tileset management
Usage: python tileset_cli.py [command] [options]
"""

import argparse
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List
import xarray as xr
from datetime import datetime

# Import the tileset manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tileset_management import MapboxTilesetManager, TilesetQueryTools
from app import create_recipe_for_netcdf, prepare_netcdf_for_mapbox_advanced

def upload_netcdf(args):
    """Upload NetCDF file to Mapbox"""
    print(f"Uploading {args.file} to Mapbox...")
    
    # Initialize manager
    manager = MapboxTilesetManager(args.token, args.username)
    
    # Create recipe if needed
    if args.recipe_file:
        with open(args.recipe_file, 'r') as f:
            recipe = json.load(f)
    else:
        recipe = create_recipe_for_netcdf(args.file, args.tileset_id, args.username)
        
        # Save recipe
        if args.save_recipe:
            recipe_path = f"recipe_{args.tileset_id}.json"
            with open(recipe_path, 'w') as f:
                json.dump(recipe, f, indent=2)
            print(f"Recipe saved to: {recipe_path}")
    
    # Process and upload
    result = manager.process_netcdf_to_tileset(args.file, args.tileset_id, recipe)
    
    if result['success']:
        print(f"✅ Successfully created tileset: {result['tileset_id']}")
        if result.get('job_id'):
            print(f"Job ID: {result['job_id']}")
    else:
        print(f"❌ Error: {result.get('error')}")
        sys.exit(1)

def update_tileset(args):
    """Update existing tileset with new data"""
    print(f"Updating tileset {args.tileset_id}...")
    
    manager = MapboxTilesetManager(args.token, args.username)
    
    # Update source
    source_id = f"{args.tileset_id}_source"
    result = manager.update_tileset_source(source_id, args.file)
    print(f"Source updated: {result}")
    
    # Republish tileset
    publish_result = manager.publish_tileset(args.tileset_id)
    job_id = publish_result.get('jobId')
    
    if job_id:
        print(f"Publishing tileset... Job ID: {job_id}")
        job_result = manager.wait_for_job(job_id)
        
        if job_result['status'] == 'success':
            print("✅ Tileset updated successfully!")
        else:
            print(f"❌ Update failed: {job_result}")

def list_tilesets(args):
    """List all tilesets"""
    manager = MapboxTilesetManager(args.token, args.username)
    
    tilesets = manager.list_tilesets(limit=args.limit)
    
    print(f"\nFound {len(tilesets)} tilesets:")
    print("-" * 80)
    
    for ts in tilesets:
        print(f"ID: {ts.get('id')}")
        print(f"  Name: {ts.get('name', 'N/A')}")
        print(f"  Created: {ts.get('created', 'N/A')}")
        print(f"  Modified: {ts.get('modified', 'N/A')}")
        print(f"  Status: {ts.get('status', 'N/A')}")
        print()

def query_tileset(args):
    """Query tileset data"""
    print(f"Querying tileset {args.tileset_id}...")
    
    manager = MapboxTilesetManager(args.token, args.username)
    
    if args.statistics:
        # Get tileset statistics
        stats = manager.query_tileset_statistics(args.tileset_id)
        print("\nTileset Statistics:")
        print(json.dumps(stats, indent=2))
    
    if args.info:
        # Get tileset info
        info = manager.get_tileset_status(args.tileset_id)
        print("\nTileset Information:")
        print(json.dumps(info, indent=2))

def query_netcdf(args):
    """Query NetCDF file locally"""
    print(f"Querying {args.file}...")
    
    ds = xr.open_dataset(args.file)
    
    # Apply temporal filter
    if args.start_time or args.end_time:
        ds = TilesetQueryTools.query_temporal_range(ds, args.start_time, args.end_time)
        print(f"Filtered to time range: {args.start_time} to {args.end_time}")
    
    # Apply spatial filter
    if args.bounds:
        bounds = [float(x) for x in args.bounds.split(',')]
        if len(bounds) == 4:
            ds = TilesetQueryTools.query_spatial_bounds(ds, *bounds)
            print(f"Filtered to bounds: {bounds}")
    
    # Compute statistics
    if args.variables:
        variables = args.variables.split(',')
    else:
        variables = None
    
    stats = TilesetQueryTools.compute_statistics(ds, variables)
    
    print("\nVariable Statistics:")
    print("-" * 60)
    for var, var_stats in stats.items():
        print(f"\n{var}:")
        for stat_name, value in var_stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    # Save filtered data if requested
    if args.output:
        ds.to_netcdf(args.output)
        print(f"\nFiltered data saved to: {args.output}")
    
    ds.close()

def delete_tileset(args):
    """Delete a tileset"""
    print(f"⚠️  About to delete tileset: {args.tileset_id}")
    
    if not args.force:
        confirm = input("Are you sure? Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Deletion cancelled.")
            return
    
    manager = MapboxTilesetManager(args.token, args.username)
    
    if manager.delete_tileset(args.tileset_id):
        print("✅ Tileset deleted successfully!")
    else:
        print("❌ Failed to delete tileset")

def create_recipe(args):
    """Create a recipe file for NetCDF data"""
    print(f"Creating recipe for {args.file}...")
    
    recipe = create_recipe_for_netcdf(args.file, args.tileset_id, args.username)
    
    # Enhance recipe with custom settings
    if args.minzoom is not None:
        recipe['layers']['default']['minzoom'] = args.minzoom
    
    if args.maxzoom is not None:
        recipe['layers']['default']['maxzoom'] = args.maxzoom
    
    if args.tile_size:
        recipe['layers']['default']['tiles']['tile_size'] = args.tile_size
    
    # Save recipe
    output_path = args.output or f"recipe_{args.tileset_id}.json"
    with open(output_path, 'w') as f:
        json.dump(recipe, f, indent=2)
    
    print(f"✅ Recipe saved to: {output_path}")
    
    # Print summary
    metadata = recipe.get('metadata', {})
    print("\nRecipe Summary:")
    print(f"  Bands: {len(recipe['layers']['default']['raster_array']['bands'])}")
    print(f"  Vector pairs: {len(metadata.get('vector_pairs', []))}")
    print(f"  Scalar variables: {len(metadata.get('scalar_variables', []))}")

def batch_upload(args):
    """Upload multiple NetCDF files"""
    print(f"Batch uploading from {args.directory}...")
    
    manager = MapboxTilesetManager(args.token, args.username)
    
    # Find all NetCDF files
    nc_files = list(Path(args.directory).glob(args.pattern))
    print(f"Found {len(nc_files)} files to upload")
    
    results = []
    for i, nc_file in enumerate(nc_files):
        print(f"\n[{i+1}/{len(nc_files)}] Processing {nc_file.name}...")
        
        # Generate tileset ID
        tileset_id = re.sub(r'[^a-zA-Z0-9_-]', '_', nc_file.stem.lower())[:32]
        
        if args.prefix:
            tileset_id = f"{args.prefix}_{tileset_id}"
        
        try:
            # Create recipe
            recipe = create_recipe_for_netcdf(str(nc_file), tileset_id, args.username)
            
            # Upload
            result = manager.process_netcdf_to_tileset(str(nc_file), tileset_id, recipe)
            
            if result['success']:
                print(f"  ✅ Success: {result['tileset_id']}")
                results.append({
                    'file': str(nc_file),
                    'tileset_id': result['tileset_id'],
                    'status': 'success'
                })
            else:
                print(f"  ❌ Failed: {result.get('error')}")
                results.append({
                    'file': str(nc_file),
                    'error': result.get('error'),
                    'status': 'failed'
                })
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                'file': str(nc_file),
                'error': str(e),
                'status': 'error'
            })
    
    # Save results
    if args.report:
        report_path = args.report
    else:
        report_path = f"batch_upload_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Batch upload complete. Report saved to: {report_path}")
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"✅ Successful: {success_count}/{len(results)}")

def main():
    parser = argparse.ArgumentParser(
        description='Mapbox Tileset Management CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a NetCDF file
  python tileset_cli.py upload data.nc --tileset-id weather_data
  
  # Update existing tileset
  python tileset_cli.py update data_new.nc --tileset-id weather_data
  
  # List all tilesets
  python tileset_cli.py list
  
  # Query NetCDF file
  python tileset_cli.py query-nc data.nc --bounds "-180,-90,180,90" --variables temp,wind_u
  
  # Create recipe file
  python tileset_cli.py recipe data.nc --tileset-id weather_data --maxzoom 12
  
  # Batch upload
  python tileset_cli.py batch-upload /data/netcdf/ --pattern "*.nc" --prefix weather
        """
    )
    
    # Global arguments
    parser.add_argument('--token', help='Mapbox access token', 
                       default=os.getenv('MAPBOX_TOKEN'))
    parser.add_argument('--username', help='Mapbox username', 
                       default=os.getenv('MAPBOX_USERNAME'))
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload NetCDF to Mapbox')
    upload_parser.add_argument('file', help='NetCDF file to upload')
    upload_parser.add_argument('--tileset-id', required=True, help='Tileset ID')
    upload_parser.add_argument('--recipe-file', help='Use custom recipe file')
    upload_parser.add_argument('--save-recipe', action='store_true', 
                              help='Save generated recipe')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update existing tileset')
    update_parser.add_argument('file', help='New NetCDF file')
    update_parser.add_argument('--tileset-id', required=True, help='Tileset ID to update')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all tilesets')
    list_parser.add_argument('--limit', type=int, default=100, help='Number of tilesets to list')
    
    # Query tileset command
    query_ts_parser = subparsers.add_parser('query', help='Query tileset information')
    query_ts_parser.add_argument('--tileset-id', required=True, help='Tileset ID')
    query_ts_parser.add_argument('--statistics', action='store_true', help='Get statistics')
    query_ts_parser.add_argument('--info', action='store_true', help='Get tileset info')
    
    # Query NetCDF command
    query_nc_parser = subparsers.add_parser('query-nc', help='Query NetCDF file')
    query_nc_parser.add_argument('file', help='NetCDF file to query')
    query_nc_parser.add_argument('--variables', help='Comma-separated list of variables')
    query_nc_parser.add_argument('--start-time', help='Start time (ISO format)')
    query_nc_parser.add_argument('--end-time', help='End time (ISO format)')
    query_nc_parser.add_argument('--bounds', help='Spatial bounds: west,south,east,north')
    query_nc_parser.add_argument('--output', help='Save filtered data to file')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a tileset')
    delete_parser.add_argument('--tileset-id', required=True, help='Tileset ID to delete')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # Recipe command
    recipe_parser = subparsers.add_parser('recipe', help='Create recipe file')
    recipe_parser.add_argument('file', help='NetCDF file')
    recipe_parser.add_argument('--tileset-id', required=True, help='Tileset ID')
    recipe_parser.add_argument('--output', help='Output recipe file path')
    recipe_parser.add_argument('--minzoom', type=int, help='Minimum zoom level')
    recipe_parser.add_argument('--maxzoom', type=int, help='Maximum zoom level')
    recipe_parser.add_argument('--tile-size', type=int, choices=[256, 512], 
                              help='Tile size')
    
    # Batch upload command
    batch_parser = subparsers.add_parser('batch-upload', help='Upload multiple files')
    batch_parser.add_argument('directory', help='Directory containing NetCDF files')
    batch_parser.add_argument('--pattern', default='*.nc', help='File pattern')
    batch_parser.add_argument('--prefix', help='Prefix for tileset IDs')
    batch_parser.add_argument('--report', help='Report output file')
    
    args = parser.parse_args()
    
    # Check credentials
    if not args.token or not args.username:
        print("❌ Error: Mapbox token and username required")
        print("Set MAPBOX_TOKEN and MAPBOX_USERNAME environment variables or use --token and --username")
        sys.exit(1)
    
    # Execute command
    if args.command == 'upload':
        upload_netcdf(args)
    elif args.command == 'update':
        update_tileset(args)
    elif args.command == 'list':
        list_tilesets(args)
    elif args.command == 'query':
        query_tileset(args)
    elif args.command == 'query-nc':
        query_netcdf(args)
    elif args.command == 'delete':
        delete_tileset(args)
    elif args.command == 'recipe':
        create_recipe(args)
    elif args.command == 'batch-upload':
        batch_upload(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()