srun -N1 -n4 --account=app-3437 --gres=gpu:1 --time=01:00:00 --pty /bin/bash -l

srun -N1 -n10 --partition=okeanos --account=NR_GRANTU --time=01:00:00 --pty /bin/bash -l