# 1) Create all needed directories if they don't already exist
@(
  "templates",
  "static\css",
  "static\js",
  "utils",
  "tests"
) | ForEach-Object {
  if ( -not (Test-Path $_) ) {
    New-Item -ItemType Directory -Path $_ | Out-Null
  }
}

# 2) Create all top-level files if they don't already exist
@(
  "app.py",
  "tileset_management.py",
  "tileset_cli.py",
  "requirements.txt",
  ".env.example",
  ".gitignore",
  "README.md",
  "Dockerfile",
  "docker-compose.yml",
  "run.py"
) | ForEach-Object {
  if ( -not (Test-Path $_) ) {
    New-Item -ItemType File -Path $_ | Out-Null
  }
}

# 3) Create all template files
@(
  "templates\index.html",
  "templates\advanced_visualization.html",
  "templates\demo.html"
) | ForEach-Object {
  if ( -not (Test-Path $_) ) {
    New-Item -ItemType File -Path $_ | Out-Null
  }
}

# 4) Create all static assets
@(
  "static\css\style.css",
  "static\js\wind-particles.js"
) | ForEach-Object {
  if ( -not (Test-Path $_) ) {
    New-Item -ItemType File -Path $_ | Out-Null
  }
}

# 5) Create utilities
@(
  "utils\__init__.py",
  "utils\recipe_generator.py",
  "utils\query_tools.py"
) | ForEach-Object {
  if ( -not (Test-Path $_) ) {
    New-Item -ItemType File -Path $_ | Out-Null
  }
}

# 6) Create tests
@(
  "tests\__init__.py",
  "tests\test_recipe_generation.py"
) | ForEach-Object {
  if ( -not (Test-Path $_) ) {
    New-Item -ItemType File -Path $_ | Out-Null
  }
}

Write-Host "Project skeleton ensured." -ForegroundColor Green
