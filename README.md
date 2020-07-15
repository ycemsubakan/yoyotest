[![Build Status](https://travis-ci.com/ycemsubakan/yoyotest.png?branch=master)](https://travis-ci.com/ycemsubakan/yoyotest)
[![codecov](https://codecov.io/gh/ycemsubakan/yoyotest/branch/master/graph/badge.svg)](https://codecov.io/gh/ycemsubakan/yoyotest)

# yoyotest


yoyotest is wonderful!


* Free software: MIT license


## Instructions to setup the project

### Install the dependencies:
(remember to activate the virtual env if you want to use one)
Add new dependencies (if needed) to setup.py.

    pip install -e .

### Add git:

    git init

### Setup pre-commit hooks:
(this will run flake8 before any commit)

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -

### Commit the code

    git add .
    git commit -m 'first commit'

### Link github to your local repository
Go on github and follow the instructions to create a new project.
When done, do not add any file, and follow the instructions to
link your local git to the remote project, which should look like this:

    git remote add origin git@github.com:ycemsubakan/yoyotest.git
    git push -u origin master

### Add Travis
A travis configuration file (`.travis.yml`) is already in your repository (so, no need to
create it). This will run `flake8` and run the tests under `tests`.

To enable it server-side, just go to https://travis-ci.com/account/repositories and click
` Manage repositories on GitHub`. Give the permission to run on the git repository you just created.

Note, the link for public project may be https://travis-ci.org/account/repositories .

### Add Codecov
Go to https://codecov.io/ and enable codecov for your repository.
If the github repository is a private one, you will need to get a
secret token for your project and add it to
github.
(see https://docs.codecov.io/docs/about-the-codecov-bash-uploader#section-upload-token)

## Running the code

### Run the tests
Just run (from the root folder):

    pytest

### Run the code/examples.
Note that the code should already compile at this point.

Running examples can be found under the `examples` folder.

In particular, you will find examples for:
* local machine (e.g., your laptop).
* a slurm cluster.

For both these cases, there is the possibility to run with or without Orion.
(Orion is a hyper-parameter search tool - see https://github.com/Epistimio/orion -
that is already configured in this project)

#### Run locally

For example, to run on your local machine without Orion:

    cd examples/local
    sh run.sh

This will run a simple MLP on a simple toy task: sum 5 float numbers.
You should see an almost perfect loss of 0 after a few epochs.

Note you have two new folders now:
* output: contains the models and a summary of the results.
* mlruns: produced by mlflow, contains all the data for visualization.
You can run mlflow from this folder (`examples/local`) by running
`mlflow ui`.

#### Run on the Mila cluster
(NOTE: this example also apply to Compute Canada - use the folders
`slurm_cc` and `slurm_cc_orion` instead of `slurm_mila` and `slurm_mila_orion`.)

To run with SLURM, on the Mila cluster, just:

    cd examples/slurm_mila
    sh run.sh

Check the log to see that you got an almost perfect loss (i.e., 0).

#### Measure GPU time (and others) on the Mila cluster

You can track down the GPU time (and other resources) of your jobs by
associating a tag to the job (when using `sbatch`).
To associate a tag to a job, replace `my_tag` with a proper tag,
and uncomment the line (i.e., remove one #) from the line:

    ##SBATCH --wckey=my_tag

This line is inside the file `examples/slurm_mila/to_submit.sh`.

To get a sumary for a particular tag, just run:

    sacct --allusers --wckeys=my_tag --format=JobID,JobName,Start,Elapsed -X -P --delimiter=','

(again, remember to change `my_tag` into the real tag name)

#### Run with Orion on the Mila cluster

This example will run orion for just one trial (see the orion config file).
Note the folder structure. The root folder for this example is
`examples/slurm_mila_orion`.
Here you can find the orion config file (`orion_config.yaml`), as well as the config
file for your project (that contains the hyper-parametersi - `config.yaml`).
Also, in this folder you will find (after running Orion) the orion db file and the
mlruns folder (i.e., the folder containing the mlflow results).

The `trial01` folder contains the executable (`run.sh`) - same for `trial02`.
The reason why there are more trial folders (`trial01` and `trial02) is because you
may want to run more trials in parallel. If you want to run more than two in paralle,
just create more trial folders, e.g.,

    cp -r trial01 trial03

and launch `run.sh` in all the trial folders. Orion will take care to sync the various 
experiments.

After the training is done, you can check orion status with the following commands:
(to be run inside any trial folder, e.g., `examples/slurm_mila_orion/trial01`)

    export ORION_DB_ADDRESS='orion_db.pkl'
    export ORION_DB_TYPE='../pickleddb'
    orion status
    orion info --name my_exp

Also, you will find the results in a folder called `orion_working_dir` (it will be
found inside `examples/slurm_mila_orion`).
In this folder, there will be a folder for very Orion trial (in this experiment, there
is just one trial - so, just one folder).
Inside you can find the models (the best one and the last one), the config file with
the hyper-parameters for this trial, and the log file.

### Building docs:

To automatically generate docs for your project, cd to the `docs` folder then run:

    make html

To view the docs locally, open `docs/_build/html/index.html` in your browser.


## YOUR PROJECT README:

* __TODO__
