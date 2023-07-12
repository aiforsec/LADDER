#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="$1"
#SBATCH --time=0-6:0:0
#SBATCH --partition=tier3
#SBATCH --account=malont
#SBATCH --mem=16g
#SBATCH --output "logs/"$1".out"
#SBATCH --error "logs/"$1".err"
#SBATCH --mail-user "ma8235@rit.edu"
#SBATCH --mail-type=ALL
#SBATCH --ntasks 1
#SBATCH --gres=gpu:v100:1

# activate venv
source /home/ma8235/attack-pattern/venv/bin/activate
python hypertuneEntityClassification.py --save-path=logs/entity_extraction/$1
deactivate
EOT
