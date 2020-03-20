###############################################################################
### PARAMETERS

DT=600
step=8

###############################################################################
### RUNNING

# outer loop, block of `step`
for i in $(seq 0 $step 31); do
	# inner loop, parallelized to complete a block
	for j in $(seq -f "%02g" $i $(($i+$step-1))); do
		python mcmc_pal5.py -i ${j} \
			  -o output/fitsigma/mwpot14-fitsigma-${j}.dat \
		      --dt=${DT} --td=10. --fitsigma -m 6 &
   done
   wait
done

###############################################################################
### END
