
# Change the version to match the dependency-definition in the pom-file
VERSION=2.0.0
CPSIGN_PATH=libs/cpsign

echo "installing cpsign version $VERSION from location $CPSIGN_PATH to local maven repo"

mvn install:install-file -Dfile=${CPSIGN_PATH} -DgroupId=com.arosbio -DartifactId=cpsign -Dversion=${VERSION} -Dpackaging=jar -DgeneratePom=true