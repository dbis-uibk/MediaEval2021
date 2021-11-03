import os

for filename in os.listdir('plans/'):
    if filename.startswith("fixed") and filename.endswith(".py"): 
         print(filename)
         os.system(f"sbatch --job-name=mediaeval-vggish --mail-user=andreas.peintner@uibk.ac.at --time=8:00:00 --mem=250G ~/jobs/single-node-titan.job 'plans/{filename}'")