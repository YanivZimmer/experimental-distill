# How to Run Training

## The Problem

If you see: `ModuleNotFoundError: No module named 'datasets'`

**Cause:** Your IDE/terminal isn't using the virtual environment.

## Solutions

### Option 1: Use Make (Recommended)

```bash
make train-local    # Handles UV environment automatically
make train-cloud
make train-mock
```

### Option 2: Use UV Directly

```bash
uv run python training/local_entry.py
uv run python training/cloud_entry.py
uv run python training/mock_entry.py
```

### Option 3: Activate Virtual Environment

```bash
# Activate venv
source .venv/bin/activate

# Then run
python training/local_entry.py
python training/cloud_entry.py
python training/mock_entry.py

# Deactivate when done
deactivate
```

### Option 4: Configure Your IDE

#### VS Code

1. Open Command Palette (`Cmd+Shift+P`)
2. Search: "Python: Select Interpreter"
3. Choose: `./.venv/bin/python`

Or create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

#### PyCharm

1. Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select: `/path/to/.venv/bin/python`

#### Cursor / Other IDEs

Look for "Python Interpreter" settings and point to `.venv/bin/python`

## Verify Setup

```bash
# Check which python
which python
# Should show: /path/to/experimental-distill/.venv/bin/python

# Check packages installed
python -c "import datasets; print('✓ datasets installed')"
python -c "import torch; print('✓ torch installed')"
```

## Quick Setup

If `.venv` doesn't exist:

```bash
# Run setup
./setup.sh

# OR manually
uv sync
```

## Common Issues

### Issue: `.venv` folder doesn't exist
**Solution:** Run `./setup.sh` or `uv sync`

### Issue: IDE still can't find packages
**Solution:**
1. Restart IDE
2. Make sure interpreter points to `.venv/bin/python`
3. Try: `uv sync` to reinstall packages

### Issue: Different Python version
**Solution:**
```bash
# Check version
python --version  # Should be 3.9+

# If wrong version, recreate venv
rm -rf .venv
uv sync
```

## Best Practices

✅ **Always use one of:**
- `make train-local`
- `uv run python training/local_entry.py`
- `source .venv/bin/activate && python training/local_entry.py`

❌ **Don't run directly:**
- `python training/local_entry.py` (without activation)

## Summary

The training scripts need packages from `.venv/`. There are three ways to ensure this:

1. **Make** - handles it automatically
2. **UV run** - uses .venv automatically
3. **Activate venv** - manual activation

Choose the method that works best for your workflow!
