# NetCDF to Mapbox Tileset Converter

A comprehensive solution for converting NetCDF weather data files to Mapbox tilesets with support for both vector fields (wind) and scalar fields (temperature, pressure, etc.). Includes advanced visualization capabilities with multi-variable display and particle animations.

## Features

- 🌡️ **Multi-Variable Support**: Handle both scalar (temperature, pressure) and vector (wind u/v) fields
- 🗺️ **RasterArray Recipes**: Automatic generation of Mapbox MTS recipes for proper band configuration
- 💨 **Wind Animation**: Real-time particle system for wind visualization
- 📊 **Advanced Queries**: Temporal and spatial filtering of data
- 🎨 **Customizable Visualizations**: Multiple color schemes and layer controls
- 🔧 **CLI Tools**: Command-line utilities for batch processing and tileset management
- 📸 **Export Capabilities**: Screenshot and GeoTIFF download options

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mapbox-netcdf-converter.git
cd mapbox-netcdf-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Mapbox credentials
```

## Quick Start

### Web Interface

1. Start the application:
```bash
python run.sh
# Or directly: uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Open http://localhost:8000 in your browser

3. Upload a NetCDF file and enable "Create Mapbox Tileset"

### Command Line Interface

Upload a single file:
```bash
python tileset_cli.py upload weather_data.nc --tileset-id weather_2024
```

Batch upload:
```bash
python tileset_cli.py batch-upload /data/netcdf/ --pattern "*.nc" --prefix weather
```

Query data:
```bash
python tileset_cli.py query-nc data.nc --bounds "-180,-90,180,90" --variables temp,wind_u,wind_v
```

## API Endpoints

### Basic Processing
- `POST /process` - Process NetCDF file with optional Mapbox upload
- `POST /process-advanced` - Advanced processing with query support

### Visualization
- `GET /visualize/{tileset_id}` - Multi-variable visualization interface
- `GET /demo` - Demo page for tileset visualization

### Data Access
- `GET /download-tif/{variable}` - Download variable as GeoTIFF
- `GET /download-recipe/{tileset_id}` - Download MTS recipe JSON
- `GET /api/tileset-info/{tileset_id}` - Get tileset metadata

### Tileset Management
- `GET /check-job/{job_id}` - Check tileset publish status
- `POST /api/query-tileset` - Query tileset statistics

## Recipe Format

The system automatically generates RasterArray recipes for Mapbox:

```json
{
  "version": 1,
  "layers": {
    "default": {
      "source": "mapbox://tileset-source/username/tileset_id",
      "minzoom": 0,
      "maxzoom": 14,
      "raster_array": {
        "bands": {
          "wind_u": {
            "band": 1,
            "source_band": "u10"
          },
          "wind_v": {
            "band": 2,
            "source_band": "v10"
          },
          "temperature": {
            "band": 3,
            "source_band": "t2m"
          }
        }
      }
    }
  }
}
```

## CLI Commands

### Upload Commands
```bash
# Upload with custom recipe
python tileset_cli.py upload data.nc --tileset-id my_data --recipe-file custom_recipe.json

# Update existing tileset
python tileset_cli.py update new_data.nc --tileset-id my_data
```

### Query Commands
```bash
# Query NetCDF with filters
python tileset_cli.py query-nc data.nc \
  --start-time "2024-01-01" \
  --end-time "2024-06-30" \
  --bounds "-130,-50,-60,50" \
  --variables "temperature,wind_u,wind_v" \
  --output filtered_data.nc

# Get tileset statistics
python tileset_cli.py query --tileset-id my_data --statistics
```

### Management Commands
```bash
# List all tilesets
python tileset_cli.py list

# Delete tileset
python tileset_cli.py delete --tileset-id old_data --force

# Create recipe only
python tileset_cli.py recipe data.nc --tileset-id my_data --maxzoom 12
```

## Visualization Features

### Multi-Variable Display
- Simultaneous display of temperature fields and wind animations
- Independent layer controls for each variable
- Customizable color schemes (thermal, viridis, plasma, cool)

### Wind Animation
- Particle-based wind flow visualization
- Adjustable particle count (1000-10000)
- Variable animation speed and trail length
- Real-time performance optimization

### Interactive Controls
- Layer toggles and opacity sliders
- Color scheme selection
- Zoom and pan controls
- Data value display at cursor position

## Data Requirements

### NetCDF Format
- Coordinate dimensions: longitude/latitude (or lon/lat)
- Optional time dimension
- Variables in float32 format
- EPSG:4326 projection

### Supported Variables
- **Scalar fields**: temperature, pressure, humidity, precipitation
- **Vector fields**: u/v wind components, ocean currents
- **Naming patterns**: u10/v10, wind_u/wind_v, water_u/water_v

## Docker Deployment

Build and run with Docker:
```bash
docker build -t mapbox-netcdf .
docker run -p 8000:8000 --env-file .env mapbox-netcdf
```

## Render.com Deployment

Deploy to Render using the included `render.yaml`:
1. Connect your GitHub repository
2. Set environment variables in Render dashboard
3. Deploy automatically on push

## Troubleshooting

### Common Issues

1. **Upload credentials error**
   - Ensure token has `tilesets:write` scope
   - Verify username is correct

2. **Recipe generation fails**
   - Check NetCDF has proper coordinate names
   - Ensure variables have valid data

3. **Visualization not loading**
   - Verify tileset publishing completed
   - Check browser console for errors

### Debug Mode

Enable debug logging:
```bash
export DEBUG=1
python run.sh
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Mapbox for the MTS API and GL JS library
- xarray community for NetCDF handling
- OpenWeather for sample data formats