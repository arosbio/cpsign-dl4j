#!/bin/bash

if [ $# -eq 0 ]; then
	# By default - skip tests when running
	mvn clean package -DskipTests=true	
else
	# Pass along all arguments to the maven build
	mvn clean package $@
fi

exit 0

# If we got arguments
if [[ $# -gt 0 ]]; then
	
	
	if [[ $# -gt 1 || ! $1 =~ (-s|--skip)$ ]]; then
		echo "build.sh only supports -s/--skip argument or no arguments at all"
		exit 1
	fi
	
	mvn clean 
	mvn help:evaluate -Dexpression=project.version -q -DforceStdout
    exit 0
fi

mvn clean -Dmaven.test.skip=true package

# remove the .jar from the final filename
#mv target/cpsign-*.jar target/cpsign
