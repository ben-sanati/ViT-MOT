root="\"/scratch/bes1g19/UG-Research/ViTDet/MAKE/\""

# Train Model
# eval sbatch --partition=ecsstudents $root"experiments_init.sh"
eval sbatch --partition=lyceum $root"experiments_init.sh"
