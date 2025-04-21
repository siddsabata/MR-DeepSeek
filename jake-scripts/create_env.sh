
# 1. Load Anaconda
module load anaconda3

# 2. Create conda env in your project space (non-interactively)
#    --yes automatically answers "yes" to prompts
conda create \
  --yes \
  --prefix /ocean/projects/cis250063p/jbentley/my_env python=3.10

# 3. Install desired libraries
#    We can either conda install or pip install, or both.
conda run -p /ocean/projects/cis250063p/jbentley/my_env \
  pip install transformers accelerate datasets peft \
  pip install -U bitsandbytes

# 4. Print a message
echo "finded creating enviroment"