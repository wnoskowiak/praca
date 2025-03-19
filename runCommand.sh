srun -N1 -n4 --account=app-3437 --gres=gpu:1 --time=01:00:00 --pty /bin/bash -l

srun -N1 -n10 --partition=okeanos --account=GS82-12 --time=01:00:00 --pty /bin/bash -l

singularity exec -B $(pwd):/workspace -B ../Inclusive_with_wire_info:/Inclusive_with_wire_info your_container.sif python /workspace/imageGenerate.py
singularity exec -B $(pwd):/workspace -B ../Inclusive_with_wire_info:/Inclusive_with_wire_info your_container.sif python /workspace/imageProcessor.py