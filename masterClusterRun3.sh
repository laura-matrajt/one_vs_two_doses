#!/bin/bash

source /app/lmod/lmod/init/profile
ml Python

#main analysis and VEI = 70:
for i in {1..5}
do
  for j in {0..3}
    do
      for k in 150
      do
        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.09 --VE_P2=0.46 --VE_S1=0.06 --VE_S2=0.49 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.2 --VE_P2=0.46 --VE_S1=0.2 --VE_S2=0.49 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.35 --VE_P2=0.46 --VE_S1=0.348 --VE_S2=0.49 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3Diff_initCond.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3Diff_initCond.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3Diff_initCond.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt

#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3Diff_initCond2.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3Diff_initCond2.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3Diff_initCond2.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt

#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3IncreasedR0.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3IncreasedR0.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3IncreasedR0.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt

#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt

#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.45 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.72 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.18 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.45 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.72 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.18 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0.14 --VE_I2=0.7 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0.35 --VE_I2=0.7 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0.56 --VE_I2=0.7 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#
#
#      done
#    done
#done



##sensitivity analysis redA = 0.3, deahts and infections
#for i in {1..5}
#do
#  for j in {0..3}
#    do
#      for k in 150
#      do
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.45 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.72 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.18 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.45 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.72 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1,3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.18 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.3" --output=twoDosesOutput/%A_%a.txt
#
#      done
#    done
#done
#
##sensitivity analysis, prevalence = 0.05 and 0.3
#for i in {1..5}
#do
#  for j in 0
#    do
#      for k in 150
#      do
#
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.3 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.3 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.3 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.05 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.05 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.05 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#      done
#    done
#done
#
#
##300K per week
#for i in {1..10}
#do
#  for j in 0
#    do
#      for k in 300
#      do
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=3 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#
#      done
#    done
#done


##frac_rec = 10
#for i in {1..5}
#do
#  for j in {0..3}
#    do
#      for k in 150
#      do
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt

#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.45 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.72 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.0 --VE_P2=0.0 --VE_S1=0.18 --VE_S2=0.9 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.45 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.72 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.18 --VE_P2=0.9 --VE_S1=0.0 --VE_S2=0.0 --VE_I1=0 --VE_I2=0 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.1 --redA=0.75" --output=twoDosesOutput/%A_%a.txt

#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.1 --VE_P2=0.66 --VE_S1=0.09 --VE_S2=0.7 --VE_I1=0.14 --VE_I2=0.7 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.26 --VE_P2=0.66 --VE_S1=0.26 --VE_S2=0.7 --VE_I1=0.35 --VE_I2=0.7 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt
#        sbatch --array=1-5 -n 1 -c 1  --time 9-0 --account=janes_h --job-name=$i --wrap="python optimalVaccineAllocation2D_masterFile3.py --VE_P1=0.44 --VE_P2=0.66 --VE_S1=0.5 --VE_S2=0.7 --VE_I1=0.56 --VE_I2=0.7 --SD=$j --numDoses=$k --myprev=0.1 --frac_rec=0.2 --redA=0.75" --output=twoDosesOutput/%A_%a.txt


      done
    done
done


