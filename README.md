# CPSign DL4J extension <!-- omit in toc -->
This repo contains an extension to [CPSign](https://arosbio.com), which adds the possibility to build DL models using the [Deeplearning4j library (DL4J)](https://deeplearning4j.konduit.ai/). The DL4J package is rather large and also has the possibility to use hardware acceleration when possible, so the user is encouraged to tweak the `pom.xml` file accordingly in case a different backend can be used in order to improve computational performance and runtime.

## Table of Contents <!-- omit in toc -->
- [Building](#building)
  - [Step 1: Install CPSign to your local maven repo](#step-1-install-cpsign-to-your-local-maven-repo)
  - [Run unit-tests](#run-unit-tests)
  - [Building an uber jar](#building-an-uber-jar)
- [Running](#running)
  - [Running from CLI](#running-from-cli)
  - [Running from Java](#running-from-java)
- [Performance notes](#performance-notes)
- [Change log](#change-log)
- [TODOs](#todos)


## Building 
This project uses Maven as build tool. CPSign comes as a uber/fat-jar including all required dependencies. The building process is thus slightly different than the standard maven build where transitive dependencies can be compared between the explicit dependencies. This could cause issues with different versions of required depencies being packaged in the final jar. If there are issues that can be traced back to this, we will also start making a 'thin CPSign' with accompanied pom-file so that the package and dependencies can follow the standard maven build flow.

### Step 1: Install CPSign to your local maven repo
Using the [maven install](https://maven.apache.org/plugins/maven-install-plugin/index.html) plugin with the goal `install:install-file` CPSign can be installed to your local maven cached repository. For convenience we've included the bash script [install_cpsign.sh](install_cpsign.sh) that does this for you, assuming that you have cpsign in the directory `libs` located in the project root directory. You may change this to suit your needs, e.g. following updated versions of CPSign. 

### Run unit-tests
To verify that everything is working as it should, run unit tests using the standard:

```
mvn test
```

### Building an uber jar
The pom file is currently configured to build the DL4J extension including all DL4J libraries and its sub-dependencies, but _excluding_ CPSign main code. Running 

```
mvn clean package
```

will compile and package the jar and put it in the [target](target) directory. You may want to include the option `-DskipTests=true` if you are sure everything is working as it should, or in case you are compiling using a different Nd4j backend which only runs on a different machine which supports the other backend. If everything should be included in the final jar, including the CPSign main code, minor adjustements can be made to the `pom.xml` - required changes are marked in the file in order to do this.

## Running 

### Running from CLI
From CPSign 2.0 the main JAR file is a 'really executable JAR', meaning that it can be run simply with `./cpsign` if correct file permissions has been set on the file. When having two separate JAR files the invocation must be altered into;
```
java -cp <path-to-cpsign>:<path-to-dl4j-extension> com.arosbio.modeling.app.cli.CPSignApp <options>
```
Depending on if your running on a Linxus/Unix or Windows system you may have to tweak the separator (`:` in this example) between the two JAR files. If you have both JAR files in your current directory and they are called `cpsign` and `cpsign-dl4j.jar` the invocation would be;
```
java -cp cpsign:cpsign-dl4j.jar com.arosbio.modeling.app.cli.CPSignApp <options>
```

An alternative approach is by tweaking the pom-file to merge both the main CPSign code with the DL4J extension, thus creating a single uber-JAR and the invocation can be simplified into;
```
java -jar cpsign-dl4j.java <options>
```
given that the property `<main.class>` is changed into `com.arosbio.modeling.app.cli.CPSignApp` which is required to run the CPSign CLI start-up class. 

### Running from Java
Including this extension in a Java project would be as simple as to include the built JAR on the class path of your own probject. 

## Performance notes
The two algorithms ([DLClassifier](src/main/java/com/arosbio/ml/dl4j/DLClassifier.java) and [DLRegressor](src/main/java/com/arosbio/ml/dl4j/DLRegressor.java)) have been set using the default values from the [Deeplearning4j trouble shooting guide](https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/troubleshooting-training). These implementations are fairly 'simple', and supports more configuration possibilities using the Java API (e.g. when it comes to strategies for the IUpdater where e.g. learning rate can be altered during training time). More complex networks will need to be implemented separately. At least these serves as a starting point.

## Change log 
**0.0.1-beta2**
- `DLClassifier` now implements `PseudoProbabilisticClassifier` interface, so that InverseProbability NCM can use these models for building CP classifier models
- Fixed bug in converting CPSign FeatureVector into INDArray, when training examples had less features then test-examples, leading to a IndexOutOfBounds exception at predict-time of these, larger, feature vectors of test-examples.

**0.0.1-beta1**
- Beta version which includes most important parameters to tweak in the DL networks. Makes two MLAlgorithms available through ServiceLoader functionality, which in term allows CPSign to pick them up and use them on the CLI. The currently tweakable parameters that can be altered through CLI is:
  - Network `width`/`depth` for all hidden layers _or_ flexibility to specify the width individually for all hidden `layers`.
  - Weight initialization (`weightInit`)
  - Batch normalization (`batchNorm`)
  - Loss function to use (`lossFunc`)
  - Activation function for hidden layers (`activation`)
  - Number of epochs to run (`numEpoch`), which is the maximum epochs to run.
  - Mini batch size (`batchSize`)
  - Fraction of all training examples used as internal test-set, used for determining early stopping of the training (`testFrac`)
  - The updater used for updating weights in the network (`updater`)
  - The optimizer used for finding the gradient in the backprop, which is used together with the updater for updating the weights (`optimizer`)
  - Specify the number of epochs to continue training the network after there is no more improvement in the loss score (`earlyStopAfter`). The calculation of the loss score can be done either on training examples or an internal test-set, which is controled by the `testFrac` parameter.
  - Weight decay, used for regularization (`weightDecay`)
  - Standard `l1` and `l2` regularization terms, to control the size of the weights
  - Both an `inputDropOut` and `dropOut` for hidden layers, which are set separately. 

## TODOs

- [x] add `.install_cpsign.sh` script for installing to user local .m2 repo
- [x] fix DLRegressor - tests/settings or what is the issue?
- [x] fix failing tests (classification original parameters, regression)
- [x] Create a first release version
