#!/bin/bash

#SBATCH -p shared # Partition to submit to
#SBATCH -J 2018lfe
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 1000 # Runtime in minutes
#SBATCH --mem=15360 # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o hostname_%j.out # Standard out goes to this file
#SBATCH -e hostname_%j.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sgomez@cfa.harvard.edu # Email to which notifications will be sent

module load python/3.6.3-fasrc01
source activate mypython
mosfit -e 2018lfe.json -F redshift 0.35 -m slsnni -N 150 -E 500 800 --local-data-only -c -S 600 -i 15000 --band-list r -qs -x radiusphot temperaturephot luminosities_out times_out -w products/2018lfe_walkers.json --no-fracking -b 0
