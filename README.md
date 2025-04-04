# README: Setting Up and Troubleshooting ngp_pl NeRF Training in Docker

This document outlines the steps taken to configure a Docker environment for running the `ngp_pl` NeRF project, troubleshoot numerous dependency and build issues, and finally execute the training script successfully.

**Goal:** Create a persistent Docker environment with specific CUDA/PyTorch versions (CUDA 11.3, PyTorch 1.11.0) suitable for the VIAT/ngp_pl project, install dependencies, build custom CUDA extensions, and run training on a Windows host with an NVIDIA GPU (RTX 3060).

## Phase 1: Initial Docker Environment Setup

1.  **Prerequisites (Host Machine - Windows):**
    *   NVIDIA Driver installed.
    *   Docker Desktop installed and configured to use the WSL 2 backend.
    *   WSL 2 installed and functional.

2.  **Create `Dockerfile`:**
    *   A `Dockerfile` was created to define the base environment. Key aspects:
        *   **Base Image:** Selected an official NVIDIA CUDA image matching the required CUDA version and a compatible OS (e.g., `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04`).
        *   **System Dependencies:** Installed essential packages using `apt-get`, including `python3.8`, `python3.8-dev`, `pip`, `git`, `build-essential`, `cmake`.
        *   **Python Setup:** Set Python 3.8 as default, upgraded pip.
        *   **Core PyTorch Installation:** Installed the **exact required PyTorch version** compatible with the CUDA base image *during the image build*. This avoids later conflicts.
            ```dockerfile
            # Example line in Dockerfile
            RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
            ```
        *   **Project Code:** Copied the `ngp_pl` project files into the `/app/ngp_pl` directory within the image.

3.  **Build the Docker Image:**
    *   Command run on the host machine (PowerShell/CMD) in the directory containing the `Dockerfile`:
        ```bash
        docker build -t nerf-env-base .
        ```
    *   Tagged the image as `nerf-env-base`.

4.  **Initial Container Run:**
    *   Started a persistent container instance from the base image, enabling GPU access and mounting the project directory.
    *   Command run on the host machine:
        ```bash
        # Replace C:\path\to\project... with your actual host path
        docker run -it --gpus all `
          --name my-nerf-session `
          -v C:\path\to\your\project\on\windows\ngp_pl:/app/ngp_pl `
          nerf-env-base bash
        ```
    *   Entered the container's interactive `bash` shell.

## Phase 2: Dependency Installation and Conflict Resolution

**Challenge:** The project required specific (often older) versions of numerous libraries listed in `requirements.txt` (referred to as `newreq.txt` during troubleshooting), which conflicted heavily with potentially newer versions installed by other dependencies or present from previous attempts.

**Strategy:** Prioritize the versions listed in the project's requirements file, especially for libraries directly interacting with PyTorch or CUDA.

**Steps Performed (Inside the Container):**

1.  **Prepare `requirements.txt`:**
    *   Ensured the `requirements.txt` (or `newreq.txt`) **did not** contain lines for `torch`, `torchvision`, or `torchaudio`, as the correct versions were already installed in the base image. These lines were commented out or removed.

2.  **Install Requirements:** Attempted to install the remaining packages.
    ```bash
    cd /app/ngp_pl
    pip install -r newreq.txt # Or requirements.txt
    ```

3.  **Conflict Resolution (Iterative Process):** Encountered multiple errors and resolved them:
    *   **`mesh_core_cython==0.0.0` Not Found:**
        *   **Cause:** Custom module, not on PyPI.
        *   **Fix:**
            *   Commented out `mesh_core_cython` in `requirements.txt`.
            *   Ensured `Cython` and `cmake` were installed (`apt-get install cmake`).
            *   Installed the rest of the requirements (`pip install -r ...`).
            *   Navigated to the local source directory for `mesh_core_cython` (likely within `ngp_pl`) and built it: `pip install .`
    *   **`mkl-fft` / `mkl-random` -> `numpy` Conflict:**
        *   **Cause:** Required `numpy` version ranges were mutually exclusive.
        *   **Fix:** Commented out `mkl-fft` and `mkl-random` lines in `requirements.txt` as they were deemed non-essential for core functionality or potentially replaceable by standard numpy/scipy.
    *   **`spacy` Build Error (`numpy>=2.0` vs Python 3.8):**
        *   **Cause:** Dependency chain (`salesforce-lavis -> spacy -> thinc -> blis -> numpy>=2.0`) incompatible with Python 3.8.
        *   **Fix:** Modified `requirements.txt` to use an older `salesforce-lavis` version compatible with Python 3.8 *or* explicitly pinned `spacy` to an older compatible version (e.g., `spacy==3.4.4`).
    *   **`timm` Conflict (`0.5.4` vs `0.4.12`):**
        *   **Cause:** Explicit requirement conflicted with `salesforce-lavis` dependency.
        *   **Fix:** Changed the explicit `timm==0.5.4` line in `requirements.txt` to `timm==0.4.12` to match the dependency.
    *   **`fsspec` `TypeError` (`Callable[list...]` vs Python 3.8):**
        *   **Cause:** Installed `fsspec` version was too new for Python 3.8's type hinting syntax.
        *   **Fix:** Downgraded `fsspec` to the version likely intended by the requirements: `pip install fsspec==2022.2.0`.
    *   **`pytorch-lightning`/`torchmetrics` `ImportError` (`_compare_version`):**
        *   **Cause:** Installed `pytorch-lightning` expected an older `torchmetrics` version.
        *   **Fix:** Downgraded `torchmetrics` to the version specified in the requirements: `pip install torchmetrics==0.7.3`.

## Phase 3: Building Custom CUDA/C++ Extensions

**Challenge:** The project relies on custom, high-performance CUDA code (`vren`, `apex`) that needs to be compiled specifically for the target environment (GPU architecture, CUDA version, PyTorch version).

**Steps Performed (Inside the Container):**

1.  **`vren` Installation:**
    *   **`ModuleNotFoundError`:** Initial check confirmed `vren` wasn't installed correctly.
    *   **Wheel Attempt:** Tried installing a pre-built `.whl` file.
    *   **`ImportError: undefined symbol`:** Encountered ABI mismatch because the wheel was built for PyTorch 1.13/CUDA 11.7, while the environment had PyTorch 1.11/CUDA 11.3.
    *   **Solution: Build from Source:**
        *   Uninstalled the wheel version (`pip uninstall vren -y`).
        *   Located the `vren` source code within the project (identified as `/app/ngp_pl/models/csrc`).
        *   Ensured build tools (`build-essential`, `cmake`, `python3.8-dev`) were installed.
        *   Modified `vren`'s `setup.py` (in `/app/ngp_pl/models/csrc`) to explicitly specify the target GPU architecture (RTX 3060 -> sm_86):
            ```python
            # In setup.py's CUDAExtension definition
            extra_compile_args={
                # ... other args ...
                'nvcc': [
                    '-O2',
                    '-gencode=arch=compute_86,code=sm_86' # Added this line
                ]
            }
            ```
        *   Attempted build: `pip install .` (from within `/app/ngp_pl/models/csrc`) -> **Failed with `Killed` error.**
        *   **Cause:** Out Of Memory (OOM) during `nvcc` compilation due to insufficient RAM allocated to the Docker container.
        *   **Fix (Requires Host Action - See Phase 4, Step 1 below):** Increased Docker Desktop memory allocation (e.g., to 16GB). Restarted the container.
        *   Retried build after increasing memory: `pip install .` -> **Succeeded.**

2.  **`apex` Installation:**
    *   **`AttributeError: module 'torch' has no attribute 'library'`:** Encountered when importing `FusedAdam`.
    *   **Cause:** The installed `apex` (likely built from the latest source) contained code checking for `torch.library`, a feature not present in PyTorch 1.11.
    *   **Solution: Build Older Version from Source:**
        *   Uninstalled the problematic Apex (`pip uninstall apex -y`).
        *   Cloned the official Apex repository (`git clone https://github.com/NVIDIA/apex /tmp/apex`).
        *   Checked out an older tag compatible with PyTorch 1.11 (identified `22.03` from March 2022 as suitable):
            ```bash
            cd /tmp/apex
            git checkout 22.03
            git clean -fdx # Clean repo
            ```
        *   Built and installed this specific older version:
            ```bash
            pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
            ```
        *   This version did not contain the problematic `torch.library` checks.

## Phase 4: Runtime Fixes and Checkpointing

**Strategy:** Preserve the working state with all dependencies installed and custom modules built before making changes that require container restarts.

**Steps Performed:**

1.  **Checkpoint Before `vren` OOM Fix:**
    *   **Problem:** Building `vren` failed due to OOM (`Killed` error), requiring increased Docker memory. Increasing memory requires restarting the container with `docker run`, losing the current state.
    *   **Action (Host Machine):**
        *   Stopped the container: `docker stop my-nerf-session`
        *   Committed the container state to a new image: `docker commit my-nerf-session nerf-env-dep-fixes:latest`
        *   Removed the old container: `docker rm my-nerf-session`
        *   Restarted a *new* container using the *committed image* and increased memory (Host action - Docker Desktop GUI -> Resources -> Memory -> Apply & Restart). The `docker run` command for this restart wasn't explicitly shown but would use the committed image name.
    *   **Action (Inside New Container):** Retried building `vren` which now succeeded.

2.  **Checkpoint Before Shared Memory (`shm`) Fix:**
    *   **Problem:** After successfully installing all dependencies and building custom modules, running `train_specific.sh` resulted in `ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)`. This requires adding `--shm-size` to `docker run`.
    *   **Action (Host Machine):**
        *   Stopped the container: `docker stop my-nerf-session`
        *   Committed the container state *with working dependencies and builds* to a final image: `docker commit my-nerf-session nerf-env-built-deps:latest`
        *   Removed the old container: `docker rm my-nerf-session`

3.  **Run Final Container with `--shm-size`:**
    *   **Action (Host Machine):** Started the final container using the last checkpoint image and adding the shared memory flag:
        ```bash
        # Replace paths/volume names as needed
        docker run -it --gpus all `
          --name my-nerf-session `
          --shm-size="2g" ` # Added shared memory
          -v C:\path\to\your\project\on\windows\ngp_pl:/app/ngp_pl `
          -v nerf-data:/data/persistent ` # Optional persistent volume
          nerf-env-built-deps:latest ` # Used the final committed image
          bash
        ```

4.  **Fix Script Line Endings:**
    *   **Problem:** Running `./train_specific.sh` gave `/bin/bash^M: bad interpreter` error.
    *   **Cause:** Windows (`\r\n`) vs Linux (`\n`) line endings in the script file copied from the host.
    *   **Fix (Inside Container):**
        *   Installed `dos2unix`: `apt-get update && apt-get install -y dos2unix`
        *   Converted the script: `dos2unix train_specific.sh`

## Phase 5: Final Execution

1.  **Navigate:** `cd /app/ngp_pl` (inside the container).
2.  **Run Script:** `./train_specific.sh`.
3.  **Result:** The script executed successfully, loaded data, ran training for both specified scenes (`airliner_01`, `barbell_02`) for 10 epochs, logged metrics, and completed without runtime errors. Output images were saved to the `results/...` directory (mapped back to the host via the volume mount). Videos were not generated due to unmet conditions in the script's logic (`--dataset_name` was `nerf`, not `nsvf`).

**Conclusion:** The process involved careful Docker environment setup, meticulous dependency version management, building custom CUDA extensions from source (sometimes requiring specific older commits), and addressing Docker-specific runtime constraints like memory and shared memory allocation. Checkpointing the container state at critical junctures was essential to avoid losing progress.
