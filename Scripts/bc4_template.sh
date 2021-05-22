#!/usr/bin/env bash
#SBATCH --job-name=job_name
#SBATCH --partition=cpu
#SBATCH --time=4-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

# Change these parameters above to suit the script. Typically --time and --mem affect the priority queue the most
# And BC4 prioritises jobs that request more nodes E.g. BC4 configs (viewed with "scontrol show config") has set "PriorityFavorSmall = No"
# To specify a specific amount of memory, add "#SBATCH --mem X(MB|GB)". Use only MB or GB, not (MB|GB). Lower means less queue time. However, leaving this
# Flag blank means you have access to ~100MB memory on the node. When you submit, if you type squeue --user <user> and 
# See "QOSMinCpuNotSatisfied" then you need to increase either nodes, ntasks-per-node or cpus-per-task

# Get rid of any modules already loaded
module purge

# Generic module containing many common libraries for most python scripts
module load languages/anaconda3/2020.02-tflow-2.2.0

# If you need custom libraries that aren't contained in the BC4 modules, you'll need to create
# A custom Anaconda env. You can do this as follows:
#	1. Go to the root directory with command: "cd ~"
#	2. Download Miniconda installer: "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
#		a. Note: You can use the anaconda installer with: "wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh"
#			 However, due to it's smaller size, miniconda is probably better for the home directory on BC4
#	3. Run "bash ~/Miniconda3-latest-Linux-x86_64.sh" (Or whatever the installer file is called that you downloaded, can check with command "ls")
#	4. Now you should have anaconda installed on your BC4 home directory. You can install custom libraries with commands like:
#		. "conda install lib"
#		. "conda install pip -> pip install lib" (Better to use conda install if you can)
#	   By default, this will download to your base env. If you want a custom environment, do "conda create --name env_name" and then do "conda activate env_name"
#	   and then download the libraries. NOTE: If you need any packages requiring the CUDA toolkit, BC4 has cuda version 10.1 e.g. cudatoolkit=10.1
#	5. Comment the "module load" line above and then uncomment the line below. This points BC4 to you anaconda dir
# source /mnt/storage/home/<user_name>/miniconda3/bin/activate base

# Run script. Note change script_name to name of your script.
python -u script_name.py

# The -u flag means "unbuffered" and just mean printing happens in real time to the output files. If the printing is weird, it's probably best to remove -u

# To run this script. Use the command "sbatch bc4_template.sh" or whatever you rename it to.

# A file called slurm-<job_id>.out will be put into the directory you ran this script from. To view the output, run "tail -f slurm-<job_id>.out"
# If you want to view the entire file in real time, then run "tail -f -n +1 slurm-<job_id>.out" instead.
