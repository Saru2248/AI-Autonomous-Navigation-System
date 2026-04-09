# Data Directory

Store any input data files here:
- Custom grid maps (JSON/CSV format)
- Pre-recorded sensor logs
- Benchmark scenario files

## Example Map Format (JSON)
```json
{
  "grid_size": 20,
  "obstacles": [[2,3],[2,4],[5,8]],
  "start": [1,1],
  "goal": [18,18]
}
```

This folder is `.gitignore`d for large files. Add small map files to version control.
