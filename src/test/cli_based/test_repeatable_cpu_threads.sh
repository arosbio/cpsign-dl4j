#!/bin/bash
#
# Copyright (C) Aros Bio AB.
#
# CPSign is an Open Source Software that is dual licensed to allow you to choose a license that best suits your requirements:
#
# 1) GPLv3 (GNU General Public License Version 3) with Additional Terms, including an attribution clause as well as a limitation to use the software for commercial purposes.
#
# 2) CPSign Proprietary License that allows you to use CPSign for commercial activities, such as in a revenue-generating operation or environment, or integrate CPSign in your proprietary software without worrying about disclosing the source code of your proprietary software, which is required if you choose to use the software under GPLv3 license. See arosbio.com/cpsign/commercial-license for details.
#

#-Dorg.bytedeco.javacpp.maxbytes=35G 
# java -cp /home/jovyan/git/cpsign-dl4j/libs/cpsign:/home/jovyan/git/cpsign-dl4j/target/cpsign-dl4j.jar com.arosbio.modeling.app.cli.CPSignApp

# Set parameters
CLEAN_UP_AFTER=false
PRECOMP_DS=../resources/data/class/precomputed.jar
RUN_CMD="java -Xmx8G -Dorg.bytedeco.javacpp.maxphysicalbytes=30G -cp /home/jovyan/git/cpsign-dl4j/libs/cpsign:/home/jovyan/git/cpsign-dl4j/target/cpsign-dl4j.jar com.arosbio.modeling.app.cli.CPSignApp"
LIC_PATH=../resources/licenses/cpsign2-develop-std.license
SEED=56781214

# First check
$RUN_CMD -V

#exit 0


# DEFINE TEST FUNCTION TO CALL SEVERAL TIMES

run_test(){
    mkdir -p tmp/

    # RUN CV, the results should hopefully change between runs
    $RUN_CMD cv \
        --seed $SEED \
        --logfile "tmp/run_$1_log.txt" \
        --license $LIC_PATH \
        -ds $PRECOMP_DS \
        --sampling-strategy Random \
        --scorer DLClassifier:width=50:nEpoch=100 \
        -rf csv \
        --ncm ProbabilityMargin
        

}

## Analyse CPU first
export BACKEND_PRIORITY_CPU=10
export BACKEND_PRIORITY_GPU=1


# Round 1 - use the default settings
run_test 1


# Round 2 - 1 thread
export OMP_NUM_THREADS=1
run_test 2

# Round 3 - 5 threads
export OMP_NUM_THREADS=5
run_test 3

# Round 4 - 10 threads
export OMP_NUM_THREADS=10
run_test 4

# Round 5 - 10 threads
export OMP_NUM_THREADS=10
run_test 4

if [[ "$CLEAN_UP_AFTER" == true ]]; then
    rm -rf tmp/
fi
