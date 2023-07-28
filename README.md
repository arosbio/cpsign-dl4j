# CPSign DL4J extension <!-- omit in toc -->
This repo contains an extension to [CPSign](https://arosbio.com), which adds the possibility to build DL models using the [Deeplearning4j library (DL4J)](https://deeplearning4j.konduit.ai/). The DL4J package is rather large and also has the possibility to use hardware acceleration when possible.

## Table of Contents <!-- omit in toc -->
- [Building](#building)
  - [Run tests](#run-unit-tests)
  - [Building a thin jar](#option-1-build-a-thin-jar)
  - [Building a fat jar](#option-2-build-a-fat-jar)
- [Running](#running)
  - [Running from Java](#running-from-java)
  - [Running from CLI](#running-from-cli)
- [Performance notes](#performance-notes)
  - [Backends](#backends)
  - [Tweaking parallization](#tweaking-parallization)
- [Change log](#change-log)
- [TODOs](#todos)


## Building 
This project uses Maven as build tool and depends on the `confai` module of [CPSign](https://github.com/arosbio/cpsign). The build specification [pom](pom.xml) is currently configured to run on CPU on OS X with M1 chip, and needs to be configured differently for different hardware, i.e. in case GPU/CUDA is available. see [ND4J Backends](https://deeplearning4j.konduit.ai/multi-project/explanation/configuration/backends) for more information. The build should be tweaked in order to fit your intended usecase, for convenience we have supplied two build profiles [thinjar](#option-1-build-a-thin-jar) and [fatjar](#option-2-build-a-fat-jar) - the former should be useful in case you wish to incorporate `cpsign-dl4j` into another piece of software, and the latter is used for using directly on the CLI. Read more in each section for greater details.

### Run unit-tests
To verify that everything is working as it should, run unit tests using the standard:

```
mvn test
```
Note that the provided data sets are very small so the acheived accuracies may seem disappointing - if you wish to see more representative results of predictive performance you should try out running on your own data. Further note that if you update the `pom.xml` for another production environment than your build machine the tests will fail due to ND4J should be missing the native code for running on your build platform. You can e.g. run tests on the `nd4j-native` and change that for e.g. CUDA before build time.

### Option 1: build a thin jar
This build profile is active by default and will only package the `cpsign-dl4j` code and none of the required dependencies, so it is intended to be used as a component into other maven builds. So this build is run by the standard `mvn package`, with any other optional arguments such as `-DskipTests`.

### Option 2: build a fat jar
This build is triggered by adding the profile `fatjar`, i.e. by running: `mvn package -P fatjar`. In contrast to the `thinjar` this build will include all required dependencies, with an additional dependency to `cpsign` (i.e. the CLI module) so that the produced jar can be run directly on the CLI. That jar file is configured to run the `CPSignApp` as main class (i.e. the same as for CPSign). 


## Running 

### Running from Java
Including this extension in a Java project would be as simple as to include the built JAR on the class path of your own probject. 

### Running from CLI
If the fat jar is built (Option 2 above), CPSign and the DL4j extension is merged into a single JAR file and the main class in the manifest file is configured to the CLI entrypoint of CPSign, running the application is thus as straightforward as;

```
java -jar <jar-name>
```

Where `jar-name` is something like: `cpsign-dl4j-[version]-fatjar.jar`.

## Performance notes
The two algorithms ([DLClassifier](src/main/java/com/arosbio/ml/dl4j/DLClassifier.java) and [DLRegressor](src/main/java/com/arosbio/ml/dl4j/DLRegressor.java)) have been set using the default values from the [Deeplearning4j trouble shooting guide](https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/troubleshooting-training). These implementations are fairly 'simple', and supports more configuration possibilities using the Java API (e.g. when it comes to strategies for the `IUpdater` where e.g. learning rate can be altered during training time). More complex networks will need to be implemented separately. At least these serves as a starting point.

### Backends 
Currently the `pom.xml` specifies the `nd4j-native` (CPU based) backend, but runtime could be greatly reduced if the user has access to a CUDA/GPU backend, see [ND4J Backends](https://deeplearning4j.konduit.ai/multi-project/explanation/configuration/backends) for more information. Note that this repo is mainly intended as proof of concept, and improvements can be likely be made at several points. 

### Tweaking parallization
DL4J and ND4J tries to create as many threads as it think is optimal for using the available hardware, if you run other jobs on the same machine you may have to set the environment variables `OMP_NUM_THREADS` to not ceate too many threads which will be detrimental for performance instead, see further info at [Deeplearning4j performance issues](https://deeplearning4j.konduit.ai/multi-project/explanation/configuration/backends/performance-issues#step-13-check-omp_num_threads-performing-concurrent-inference-using-cpu-in-multiple-threads-simultan).

## Change log 

**0.0.1-beta10**
- Update Deeplearning4j version from `1.0.0-M1.1` to `1.0.0-M2.1`.
- Update to CPSign `2.0.0-rc4`, now accessible from Maven central so no need to install locally - remove the `install_cpsign.sh` script for doing that.
- Revised the build process by introducing two build profiles; thinjar and fatjar. 
- Updated nd4j backend to run on OS X M1 chip (for users to update).


**0.0.1-beta9**
- Save labels from DLClassifier if needed, e.g. custom labels that are not 0,1,2,.. This means that old models will be loaded successfully given that they have default labels, and will give strange predictions if they were something else. Added a gihub issue to DL4J repo that `labels` are not serialized, hopefully this can be solved upstream.
- Solved bug in `predictClass` of `DLClassifier` so that it works for non-standard labels, e.g. having [-1, 1] will now give correct output.

**0.0.1-beta8**
- Minor updates, e.g. remove duplicate `seed` and use the one already in `NeuralNetConfiguration.Builder` class. Add try-catch for releasing resources etc.

**0.0.1-beta7**
- Fix bug for `iterationTimeout`, missed copying over that parameter when calling `clone()` on the instances.

**0.0.1-beta6**
- Put version from pom file into `cpsign-dl4j.properties` file in the final jar, and the build timestamp as well.
- Display the version and build timestamp when running the main class of the jar
- add parameter `iterationTimeout` to set a different threshold in allowed minutes for each iteration to take before terminating training.

**0.0.1-beta5**
- Add build version and timestamp to cpsign-dl4j.properties file at build time and retreive it from the JAR main-class, allowing to get the version of a JAR file by simply running `java -jar cpsign-dl4.jar` from the CLI.
- Allow to set a custom iteration timeout instead of the default 20 minutes.

**0.0.1-beta4**
- Changed syntax for `updater` to instead use `;` so it will work both for specifying to the `--scorer` flag and to `--grid` options. Using `:` is problematic with `--scorer` as it is instead for the sub-sub-parameter and not the first order (i.e. an argument of DLClassifier or DLRegressor classes). This also requires CPSign main code of version 2.0.0-beta2 or greater. 
- Added better way to write training scores, these can now be printed to a user-specified file (`trainOutput`). Also computes the training loss scores in case an internal test-score is used for determining early stopping. 
- Found issue regarding batch size, where re-using the same batch size for the internal test-set could be lead to no test batches being passed during evaluation in each epoch - giving a score of NaN and failed training. Now uses the same batch size in case at least 2 full batches can be sent, otherwise passes all test-examples at once. 
- Added possibility for gradient normalization (`gradNorm`)
- Set higher log-level for some DL4J / ND4J internal classes to make the cpsign.log less verbose.

**0.0.1-beta3**
- The syntax for specifying `updater` now uses `:` instead of `,` to separate the implementation type with its sub-parameters. E.g. specifying Nesterovs using learning rate 0.05 is written like: `Nesterovs:0.05`. Solves issues when specifying several updaters to the grid of searched parameters.

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

- [x] ~~add `.install_cpsign.sh` script for installing to user local .m2 repo~~ this is no longer used, cpsign is accessible from Maven Central.
- [x] fix DLRegressor - tests/settings or what is the issue?
- [x] fix failing tests (classification original parameters, regression)
- [x] Create a first release version
