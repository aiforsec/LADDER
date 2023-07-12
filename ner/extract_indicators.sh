#!/bin/bash
sbatch <<EOT
#!/bin/bash


#SBATCH --job-name="$1"
#SBATCH --time=0-3:0:0
#SBATCH --partition=tier3
#SBATCH --account=malont
#SBATCH --mem=16g
#SBATCH --output "150/"$1".out"
#SBATCH --error "150/"$1".err"
#SBATCH --mail-user ma8235@rit.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks 1



# activate venv
source /home/ma8235/cyner/venv/bin/activate

python3 infer.py

deactivate

EOT
