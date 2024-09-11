# GenDexGrasp Tests using IsaacGym 
A forked version of the GenDexGrasp paper [repository](https://github.com/tengyu-liu/GenDexGrasp?tab=readme-ov-file) used for evaluating grasp generation methods. 

## Installation
- Download and install the Isaac Gym Preview 4 release from [Isaac Gym](https://developer.nvidia.com/isaac-gym), follow the installation steps to create a new conda environment (will simplify things).
- Install the requirements listed in requirements.yaml within the Isaac Gym conda environment using the following command:
```Shell
    conda install --file requirements.yaml
```

## Running tests
The run_grasp_test.py script is used to run the grasping tests described in the [GenDexGrasp paper](https://arxiv.org/abs/2210.00722), it is a slightly altered version of their script. It takes the following arguments:
- robot_name: name of the gripper to be used 
- data_dir: folder path containing all the grasps.pt files to test.
- object_list: .json file path with list of objects to evaluate (just uses the objects under the 'validate' dictionary key).
- output_dir: directory save the results to.
- output_name: name of .json file where results will be printed.
- filtered: Boolean argument to signal if .pt contain all the generated grasps for an object gripper pair or if they have already been filtered (using minimum energy).
- headless: Boolean argument to run the simulation headless.

An example run command is shown below:
```Shell
    python run_grasp_test.py --robot_name=shadowhand --data_dir=/home/felipe/Downloads/fullrobots-sharp_lift/ood-shadowhand-euclidean_dist-gen --output_dir=/home/felipe/Documents/GenDexGrasp/test --object_list=/home/felipe/Documents/GenDexGrasp/split_train_validate_objects.json --filtered --headless --output_name=ood_shadowhand_2
```

```sh
python run_grasp_test.py --output_dir=/home/felipe/Documents/GenDexGrasp/results --object_list=/home/felipe/Documents/GenDexGrasp/split_train_validate_objects.json --filtered --headless --robot_name=ezgripper --data_dir=/home/felipe/Downloads/fullrobots-sharp_lift_numg_64/ood/ezgripper --output_name=ez1

python run_grasp_test.py --output_dir=/home/felipe/Documents/GenDexGrasp/results --object_list=/home/felipe/Documents/GenDexGrasp/split_train_validate_objects.json --filtered --headless --robot_name=barrett --data_dir=/home/felipe/Downloads/fullrobots-sharp_lift_numg_64/ood/barrett --output_name=bh1

python run_grasp_test.py --output_dir=/home/felipe/Documents/GenDexGrasp/results --object_list=/home/felipe/Documents/GenDexGrasp/split_train_validate_objects.json --filtered --headless --robot_name=shadowhand --data_dir=/home/felipe/Downloads/fullrobots-sharp_lift_numg_64/ood/shadowhand --output_name=sh1
```



## Ablations

Steps:

- `cd ~/Documents/GenDexGrasp && conda activate rlgpu`
- opened a `screen` session with `>3` terminals
- t1, t2, t3: shadow, ez, barrett
- Update Adam Config
- Run t1, t2 and then once t2 is done, do t3
- Move results into folder `ablations`

**Configs**: `{model_type, learning_rate, step_size}`

CHANGE ADAM CONFIG BEFORE RUNNING ALL THREE!!!!


## FullRobots, Pen.Weight = 60, NumG = 64

Result format: `(success_rate, std, mean(std_q))`

- [x] A-01: gcs, 0.1, 0.02
- [x] A-02: gcs, 0.1, 0.05
- [x] A-03: gcs, 0.1, 0.1
- [x] A-04: gcs, 0.5, 0.02
- [x] A-05: gcs, 0.5, 0.05
- [x] A-06: gcs, 0.5, 0.1
- [x] A-07: gdx, 0.1, 0.02
- [x] A-08: gdx, 0.1, 0.05
- [x] A-09: gdx, 0.1, 0.1
- [x] A-10: gdx, 0.5, 0.02
- [x] A-11: gdx, 0.5, 0.05
- [x] A-12: gdx, 0.5, 0.1
