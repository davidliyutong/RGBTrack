# MVSDK Python Library

## Get started

### Windows

- Install the mvsdk via the official installer
- `pip install .`

### Linux

- `pip install .`
- Launch a Python terminal with **sudo/root privileges**, run:

    ```python
    from mvsdk.utils import install_driver
    install_driver()
    ```

  **Note:** Driver installation requires root privileges to write USB rules to `/etc/udev/rules.d/`
  
  After installation, you may need to restart the system or reload udev rules:
  ```bash
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

## How to remove

### Windows

- Remove with the official uninstaller

### Linux

- **First**, launch a Python terminal with **sudo/root privileges**, run:

    ```python
    from mvsdk.utils import uninstall_driver
    uninstall_driver()
    ```

  **Note:** Driver uninstallation requires root privileges to remove USB rules from `/etc/udev/rules.d/`

- **Then**, uninstall the package:
  ```bash
  pip uninstall mvsdk
  ```

## Development Installation

For development, you can install in editable mode:

```bash
pip install -e .
```

## Building Distribution

To build the package for distribution:

```bash
pip install build
python -m build
```

This will create distribution files in the `dist/` directory.

## How to use

Refer to the official guide.
