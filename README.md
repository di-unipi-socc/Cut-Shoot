# Cut&Shoot
Cut&Shoot is a tool to execute quantum circuits by applying *circuit cutting* and *shot-wise distribution* in pipeline.
![Cut&Shoot pipeline](https://github.com/di-unipi-socc/Cut-Shoot/blob/main/cutnshoot.png?raw=true)

The pipeline consists of four main steps:
1. **Cut:** The original circuit is divided into smaller fragments using the chosen *cutting tool*. The tool outputs all possible fragment variations, which are then treated as standalone quantum circuits with their respective shot allocation determined by a custom *allocation policy*.
2. **Split:** The shots for each fragment variation are distributed across the target NISQ devices according to the selected *split policy*. The shots of each fragment variation are executed independently and concurrently on the target QPUs.
3. **Merge:** The execution results from the NISQ devices are merged to obtain the probability distribution for each fragment through the user-specified *merge policy*, which may differ from the *split policy*.
4. **Sew:** Finally, the probability distributions of the fragments are combined to reconstruct the original circuit's probability distribution.

## Installation
Cut&Shoot can be installed using poetry:
```bash
git clone [this repository]
cd Cut-Shoot
poetry shell
poetry install
```

## Usage
Cut&Shoot can be used as a Python library. The following example shows how to use Cut&Shoot as a Python library:
```python
from run import run

run(
    circuit,
    shots_assignment,
    observable_string,
    cut_tool,
    provider_backend_couple,
    split_func,
    merg_func,
    cutting = True,
    shot_wise = True,
    metadata,
)
```
passing the following parameters:
- `circuit_qasm`: the quantum circuit in QASM format.
- `shots_assignment`: the policy to assign shots to each fragment.
- `observable_string`: a string representing the observable to measure.
- `cut_tool`: the cutting tool to use and its parameters.
- `provider_backend_couple`: a list of tuples, each containing a provider and a backend.
- `split_func`: the policy to split shots across the QPUs.
- `merge_func`: the policy to merge the results from the QPUs.
- `cutting`: a boolean indicating whether to perform the cutting step.
- `shot_wise`: a boolean indicating whether to perform the shot-wise distribution step.
- `metadata`: a dictionary containing additional information.

## Run the experiments
To run the experiments, execute the following command inside the ./exp folder:
```bash
python exp.py run [config_file]
```
where `config_file` is the path to the configuration file.
