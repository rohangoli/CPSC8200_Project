# CPSC8200_Project
Implementation of distributed SGD with NVSHMEM

## Find Multiple Node hostnames
cat $PBS_NODEFILE | sort | uniq