# neural operator playground

A repository for learning, testing, and experimenting with **Neural Operators (NO)**, including Fourier Neural Operator (FNO) and other models.

Mostly using [`NeuralOperator`](https://neuraloperator.github.io/dev/index.html) library, which is a PyTorch-based library for building and training neural operators.

## Pre-requisites

### Python env on HPC

load python 3.9.x:

```bash
ml load python/3.9.15
```

create a python virtual environment of 3.9.x named `neuralop`

```bash
python -m venv neuralop
```

activate python venv

```bash
source neuralop/bin/activate
```

### Install `neuraloperator` form source

download
```bash
git clone https://github.com/neuraloperator/neuraloperator.git
cd neuraloperator
```

pip install (request an interactive session)
```bash
interact -n 1 -c 1 -p install -t 30
source /path/to/neuralop/bin/activate
pip install -r requirements.txt
pip install -e .
```

also remember to install jupyter notebook in our venv
```bash
source /path/to/neuralop/bin/activate 
pip install notebook
```

## Run Jupyter notebook

Example slurm job script (open a server)

```bash
#!/bin/bash -l

#SBATCH --job-name=jupyter
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --output=jupyter-notebook-%J.log
#SBATCH --partition=parallel
#SBATCH -A sghosh20

ml python/3.9.15
source /path/to/neuralop/bin/activate

XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
echo $port
node=$(hostname -s)
user=$(whoami)
jupyter-notebook --no-browser --port=${port} --ip=${node}
```
- look at the file "jupyter-notebook-JOBID.log" for information on
  new ssh command from a different windows in your local machine (`ssh -N -L ${port}:{node}:${port} ${user}@login.rockfish.jhu.edu`)

- we need to use this ssh info to connect in a new local terminal, e.g.:
  `ssh -N -L 8251:c228:8251 yliu664@login.rockfish.jhu.edu`

- Then we can open the server link: 
  the line to copy and paste in your browser (`http://127.0.0.:PORT-number/?token...`)
