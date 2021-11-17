# cpsign-dl4j
This repo contains an extension to CPSign which adds the possibility to build DL models using the DL4J library.  The DL4J package is rather large and also has the possibility to use hardware acceleration when possible, so the user is encouraged to tweak the `pom.xml` file accordingly in case a different backend can be used in order to improve computational performance and runtime.

## Building 
This project uses Maven as build tool. CPSign comes as a uber/fat-jar including all required dependencies. The building process is thus slightly different than the standard maven process where transitive dependencies can be compared between the explicit dependencies. This could cause issues with different versions of required depencies being packaged in the final jar. If there are issues that can be traced back to this, we will also start making a 'thin CPSign' with accompanied pom-file so that the package and dependencies can follow the standard maven flow.

### Install CPSign to your local maven repo
Using the [maven install](https://maven.apache.org/plugins/maven-install-plugin/index.html) plugin with the goal `install:install-file` the file can be installed to your local maven cached repository. For convenience we've included the bash script [install_cpsign.sh](install_cpsign.sh) that does this for you, assuming that you have cpsign in the directory `libs` located in the project root directory. You may change this to suit your needs. 

### Run unit-tests
To verify that everything is working as it should, run unit tests using the standard:

```
mvn test
```

### Building an uber jar
The pom file is currently configured to build the deeplearning4j extension including all DL4J libraries and its sub-dependencies, but _excluding_ CPSign. Running 

```
mvn clean package
```

will compile and package the jar and put it in the [target](target) directory. You may want to include the `-DskipTests=true` parameter if you are sure everything is working as it should, or in case you are compiling using a different Nd4j backend which only runs on a different machine which supports the other backend. If everything should be included in the final jar, including the CPSign main code, minor adjustements can be made to the `pom.xml`.

## Performance notes
The two algorithms ([DLClassifier](src/main/java/com/arosbio/ml/dl4j/DLClassifier.java) and [DLRegressor](src/main/java/com/arosbio/ml/dl4j/DLRegressor.java)) have been set using the default values from the [Deeplearning4J trouble shooting guide](https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/troubleshooting-training). These implementations are fairly 'simple', and supports more configuration possibilities using the Java API (e.g. when it comes to strategies for the IUpdater where e.g. learning rate can be altered during training time). More complex networks will need to be implemented separately. At least these serves as a starting point.

## TODOs

- [x] add `.install_cpsign.sh` script for installing to user local .m2 repo
- [x] fix DLRegressor - tests/settings or what is the issue?
- [x] fix failing tests (classification original parameters, regression)
- [ ] Split tests into Integration Tests (*IT) and unit-tests. Include other testing-dep.
- [ ] Create a first release version
