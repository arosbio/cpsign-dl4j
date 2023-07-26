package com.arosbio.ml.dl4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.arosbio.chem.io.in.SDFile;
import com.arosbio.cheminf.data.ChemDataset;
import com.arosbio.cheminf.descriptors.DescriptorFactory;
import com.arosbio.cheminf.descriptors.fp.ECFP4;
import com.arosbio.cheminf.io.ModelSerializer;
import com.arosbio.commons.FuzzyServiceLoader;
import com.arosbio.commons.GlobalConfig;
import com.arosbio.cpsign.app.CPSignApp;
import com.arosbio.cpsign.app.ExplainArgument;
import com.arosbio.cpsign.app.Train;
import com.arosbio.cpsign.app.TuneScorer;
import com.arosbio.cpsign.app.utils.CLIConsole;
import com.arosbio.cpsign.app.utils.CLIConsole.VerbosityLvl;
import com.arosbio.data.DataRecord;
import com.arosbio.data.DataUtils;
import com.arosbio.data.Dataset;
import com.arosbio.data.Dataset.SubSet;
import com.arosbio.data.NamedLabels;
import com.arosbio.data.transform.ColumnSpec;
import com.arosbio.data.transform.feature_selection.DropColumnSelector;
import com.arosbio.data.transform.scale.RobustScaler;
import com.arosbio.data.transform.scale.Standardizer;
import com.arosbio.ml.ClassificationUtils;
import com.arosbio.ml.algorithms.MLAlgorithm;
import com.arosbio.ml.cp.acp.ACPClassifier;
import com.arosbio.ml.cp.nonconf.classification.InverseProbabilityNCM;
import com.arosbio.ml.io.ModelInfo;
import com.arosbio.ml.metrics.Metric;
import com.arosbio.ml.metrics.classification.BalancedAccuracy;
import com.arosbio.ml.metrics.classification.ClassifierAccuracy;
import com.arosbio.ml.nd4j.ND4JUtil.DataConverter;
import com.arosbio.ml.sampling.RandomSampling;
import com.arosbio.ml.testing.RandomSplit;
import com.arosbio.ml.testing.TestRunner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Range;

import test_utils.UnitTestBase;

public class TestDL4JClassifier extends UnitTestBase {

	@Test
	public void testTrainSaveAndLoad_StandardLabels() throws IllegalArgumentException, IOException {
		GlobalConfig.getInstance().setRNGSeed(56789);
		DLClassifier clf = new DLClassifier();
		clf.numEpoch(200) //1000
			.testSplitFraction(0)
			.numHiddenLayers(3)
			.batchSize(-1)
			.updater(new Sgd(0.1))
			.gradientNorm(GradientNormalization.ClipL2PerLayer)
			.activation(Activation.TANH);
		
		SubSet allData = getIrisClassificationData();

		SubSet[] dataSplits = allData.splitRandom(.7);
		SubSet trainingData = dataSplits[0];
		SubSet testData = dataSplits[1];
		
		// Standardize training data and transform test-data as well
		Standardizer std = new Standardizer();
		trainingData = std.fitAndTransform(trainingData);
		testData = std.transform(testData);

		// Find the order of the labels..
		Set<Integer> labelsSet = new HashSet<>();
		for (DataRecord r : trainingData){
			if (! labelsSet.contains((int)r.getLabel())){
				labelsSet.add((int)r.getLabel());
			}
		}
		// Verify that labels are the standard ones
		Assert.assertEquals(ImmutableSet.of(0,1,2), labelsSet);
		
		// Train it
		clf.train(trainingData);

		// System.err.println("Labels after training: " + clf.getLabels());

		// Evaluate the first one
		BalancedAccuracy ba = new BalancedAccuracy();
		ClassifierAccuracy ca = new ClassifierAccuracy();
		Assert.assertEquals(new HashSet<>(clf.getLabels()), labelsSet);
		// System.err.println("trained labels: " + clf.getLabels());
		
		Set<Integer> predictedClasses = new HashSet<>();
		for (DataRecord r : testData) {
			int pred = clf.predictClass(r.getFeatures());
			predictedClasses.add(pred);
			ba.addPrediction((int)r.getLabel(),pred);
			ca.addPrediction((int)r.getLabel(),pred);
		}
		// System.err.printf("BA: %s Acc: %s%n",ba.getScore(),ca.getScore());
		Assert.assertEquals(labelsSet, predictedClasses);
		// System.err.println("Predicted classes: " + predictedClasses);
		// Save the model
		File modelFile = File.createTempFile("model", ".net"); 
		try (OutputStream ostream = new FileOutputStream(modelFile)){
			clf.saveToStream(ostream);
		}
		
		// Load it from file
		DLClassifier loaded = new DLClassifier();
		try (InputStream istream = new FileInputStream(modelFile);){
			loaded.loadFromStream(istream);
		}
		Assert.assertEquals(clf.getLabels(), loaded.getLabels());
		
		BalancedAccuracy ba2 = new BalancedAccuracy();
		ClassifierAccuracy ca2 = new ClassifierAccuracy();
		predictedClasses.clear();
		for (DataRecord r : testData) {
			int pred = loaded.predictClass(r.getFeatures());
			predictedClasses.add(pred);
			ba2.addPrediction((int)r.getLabel(),pred);
			ca2.addPrediction((int)r.getLabel(),pred);
		}
		Assert.assertEquals(labelsSet, predictedClasses);
		// System.err.println("Predicted classes (loaded): " + predictedClasses);
		// Check that the loaded model produces the same results
		Assert.assertEquals(ba.getScore(), ba2.getScore(), 0.000001);
		Assert.assertEquals(ca.getScore(), ca2.getScore(), 0.000001);
		
		clf.releaseResources();
		loaded.releaseResources();
		
	}

	@Test
	public void testTrainSaveAndLoad() throws IllegalArgumentException, IOException {
		GlobalConfig.getInstance().setRNGSeed(56789);
		DLClassifier clf = new DLClassifier();
		clf.numEpoch(200) //1000
			.testSplitFraction(0)
			.numHiddenLayers(3)
			.batchSize(-1)
			.updater(new Sgd(0.1))
			.gradientNorm(GradientNormalization.ClipL2PerLayer)
			.activation(Activation.TANH);
		
		SubSet allData = getIrisClassificationData();

		// Create "custom" labels -1, 1, 2 to make sure we serialize/load labels correctly
		for (DataRecord r : allData){
			if (r.getLabel() == 0){
				r.setLabel(-1);
			}
		}


		SubSet[] dataSplits = allData.splitRandom(.7);
		SubSet trainingData = dataSplits[0];
		SubSet testData = dataSplits[1];
		
		// Standardize training data and transform test-data as well
		Standardizer std = new Standardizer();
		trainingData = std.fitAndTransform(trainingData);
		testData = std.transform(testData);

		// Find the order of the labels..
		Set<Integer> labelsSet = new HashSet<>();
		for (DataRecord r : trainingData){
			if (! labelsSet.contains((int)r.getLabel())){
				// System.err.println("New label: " + r.getLabel());
				labelsSet.add((int)r.getLabel());
			}
		}
		
		// Train it
		clf.train(trainingData);

		// System.err.println("Labels after training: " + clf.getLabels());

		// Evaluate the first one
		BalancedAccuracy ba = new BalancedAccuracy();
		ClassifierAccuracy ca = new ClassifierAccuracy();
		Assert.assertEquals(new HashSet<>(clf.getLabels()), labelsSet);
		// System.err.println("trained labels: " + clf.getLabels());
		
		Set<Integer> predictedClasses = new HashSet<>();
		for (DataRecord r : testData) {
			int pred = clf.predictClass(r.getFeatures());
			predictedClasses.add(pred);
			ba.addPrediction((int)r.getLabel(),pred);
			ca.addPrediction((int)r.getLabel(),pred);
		}
		// System.err.printf("BA: %s Acc: %s%n",ba.getScore(),ca.getScore());
		Assert.assertEquals(labelsSet, predictedClasses);
		// System.err.println("Predicted classes: " + predictedClasses);
		// Save the model
		File modelFile = File.createTempFile("model", ".net"); 
		try (OutputStream ostream = new FileOutputStream(modelFile)){
			clf.saveToStream(ostream);
		}
		
		// Load it from file
		DLClassifier loaded = new DLClassifier();
		try (InputStream istream = new FileInputStream(modelFile);){
			loaded.loadFromStream(istream);
		}
		Assert.assertEquals(clf.getLabels(), loaded.getLabels());
		
		BalancedAccuracy ba2 = new BalancedAccuracy();
		ClassifierAccuracy ca2 = new ClassifierAccuracy();
		predictedClasses.clear();
		for (DataRecord r : testData) {
			int pred = loaded.predictClass(r.getFeatures());
			predictedClasses.add(pred);
			ba2.addPrediction((int)r.getLabel(),pred);
			ca2.addPrediction((int)r.getLabel(),pred);
		}
		Assert.assertEquals(labelsSet, predictedClasses);
		// System.err.println("Predicted classes (loaded): " + predictedClasses);
		// Check that the loaded model produces the same results
		Assert.assertEquals(ba.getScore(), ba2.getScore(), 0.000001);
		Assert.assertEquals(ca.getScore(), ca2.getScore(), 0.000001);
		
		clf.releaseResources();
		loaded.releaseResources();
		
	}


	// @Test
	public void testNumBytes() throws Exception {
		// Simple check to see appropriate number of bytes to mark in the loading of labels
		System.err.println(""+ "[12,12,125,21,24,12,521,125,51,215,152]".getBytes("UTF-8").length);
	}

	@Test
	public void testSaveLoadCPSignModel() throws Exception {
		GlobalConfig.getInstance().setRNGSeed(56789);
		DLClassifier clf = new DLClassifier();
		clf.numEpoch(200) //1000
			.testSplitFraction(0)
			.numHiddenLayers(3)
			.batchSize(-1)
			.updater(new Sgd(0.1))
			.gradientNorm(GradientNormalization.ClipL2PerLayer)
			.activation(Activation.TANH);
		
		SubSet allData = getBinaryClassificationData(); //getIrisClassificationData();
		// Set custom labels [-1, 1]
		Set<Double> labels = new HashSet<>();
		for (DataRecord r : allData){
			if (r.getLabel() == 0){
				r.setLabel(-1);
			}
			labels.add(r.getLabel());
		}
		Assert.assertEquals(ImmutableSet.of(-1d,1d), labels);

		SubSet[] dataSplits = allData.splitRandom(.7);
		SubSet trainingData = dataSplits[0];
		SubSet testData = dataSplits[1];

		
		// Standardize training data and transform test-data as well
		Standardizer std = new Standardizer();
		// allData = std.fitAndTransform(allData);
		trainingData = std.fitAndTransform(trainingData);
		testData = std.transform(testData);

		// Create ACP implementation
		ACPClassifier acp = new ACPClassifier(new InverseProbabilityNCM(new DLClassifier()),new RandomSampling());

		// Create ChemPredictor
		
		
		// Wrap SubSet in Dataset
		Dataset data = new Dataset();
		data.withDataset(trainingData);

		// Train ACP
		acp.train(data);
		// System.err.println("ACP props: " + acp.getProperties());

		// Evaluate the first one
		BalancedAccuracy ba = new BalancedAccuracy();
		ClassifierAccuracy ca = new ClassifierAccuracy();
		Set<Integer> predictedClasses = new HashSet<>();
		
		for (DataRecord r : testData) {
			Map<Integer,Double> pvals = acp.predict(r.getFeatures());
			int predClass = ClassificationUtils.getPredictedClass(pvals);
			predictedClasses.add(predClass);
			// int pred = clf.predictClass(r.getFeatures());
			ba.addPrediction((int)r.getLabel(),predClass);
			ca.addPrediction((int)r.getLabel(),predClass);
		}
		Assert.assertEquals(ImmutableSet.of(-1,1), predictedClasses);

		// System.err.printf("BA: %s Acc: %s%n",ba.getScore(),ca.getScore());
		
		// Save the model
		File modelFile = File.createTempFile("model", ".net");
		acp.setModelInfo( new ModelInfo("name"));
		ModelSerializer.saveModel(acp, modelFile, null); //predictor, jar, spec);


		// LoggerUtils.setDebugMode(System.err);
		// Load the model again
		ACPClassifier loaded = (ACPClassifier) ModelSerializer.loadPredictor(modelFile.toURI(), null);

		// System.err.println("loaded properties: " + loaded.getProperties());
		
		
		BalancedAccuracy ba2 = new BalancedAccuracy();
		ClassifierAccuracy ca2 = new ClassifierAccuracy();
		predictedClasses.clear();
		for (DataRecord r : testData) {
			Map<Integer,Double> pvals = loaded.predict(r.getFeatures());
			int pred = ClassificationUtils.getPredictedClass(pvals);
			predictedClasses.add(pred);
			// int pred = loaded.predictClass(r.getFeatures());
			ba2.addPrediction((int)r.getLabel(),pred);
			ca2.addPrediction((int)r.getLabel(),pred);
		}
		Assert.assertEquals(ImmutableSet.of(-1,1), predictedClasses);
		// System.err.printf("BA: %s Acc: %s%n",ba2.getScore(),ca2.getScore());

		// Check that the loaded model produces the same results
		Assert.assertEquals(ba.getScore(), ba2.getScore(), 0.000001);
		Assert.assertEquals(ca.getScore(), ca2.getScore(), 0.000001);
		
		acp.releaseResources();
		loaded.releaseResources();
	}
	
	@Test
	public void trainWithDifferentTestBatchFrac() throws IllegalArgumentException, IOException {
		// Create a tmp file to write scores to
		File tmpScoresFile = File.createTempFile("scores", ".csv");
		tmpScoresFile.deleteOnExit();

		DLClassifier clf = new DLClassifier();
		clf.numEpoch(50) // don't care if it's good or not, do not let it run to optimal weights
				.testSplitFraction(0.1) // 150 all together, 10% internal test --> 15 test examples
				.numHiddenLayers(3)
				.batchSize(16) // set batch to 16 to not get a single batch for the test-set
				.updater(new Sgd(0.1))
				.lossOutput(tmpScoresFile.toString())
				.gradientNorm(GradientNormalization.ClipL2PerLayer)
				.activation(Activation.TANH);

		SubSet allData = getIrisClassificationData();

		// Standardize training data
		Standardizer std = new Standardizer();
		std.fitAndTransform(allData);

		// Train it
		clf.train(allData);

		DLClassifier clf2 = clf.clone();
		clf2.testSplitFraction(.2)
				.batchSize(15); // 20% test-split should be 30 examples, and batch size 15 will be 2 batches

		clf2.train(allData);
		
		clf.releaseResources();
		clf2.releaseResources();
		
//		String scores = IOUtils.toString(tmpScoresFile.toURI(), StandardCharsets.UTF_8);
//		System.err.println(scores);

	}

	@Test
	public void testUseSameSettings() throws Exception {
		NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder()
				.activation(Activation.TANH)
				.weightInit(WeightInit.XAVIER)
				.updater(new Sgd(0.1))
				.l2(1e-4);

		DLClassifier clf = new DLClassifier(config);
		clf.numEpoch(100).testSplitFraction(0).numHiddenLayers(3).batchSize(-1).evalInterval(10);

		SubSet allData = getIrisClassificationData();
		allData.shuffle();

		allData = new Standardizer().fitAndTransform(allData);
		DataRecord testRec = allData.remove(0); // A single test-example

		// LoggerUtils.setDebugMode(System.out);
		clf.train(allData);
		
		Assert.assertEquals(testRec.getLabel(), ClassificationUtils.getPredictedClass(clf.predictScores(testRec.getFeatures())),0.0);
		// System.err.println("test label: "+testRec.getLabel());
		// System.err.println("Scores: "+ clf.predictScores(testRec.getFeatures()));
		clf.releaseResources();

	}

	// @Test
	public void testClassifyIris() throws Exception {
		SubSet data = getIrisClassificationData();
		data.shuffle();
		data = new Standardizer().fitAndTransform(data);

		DataConverter conv = DataConverter.classification(data);

		DataSet allData = new DataSet(conv.getFeaturesMatrix(), conv.getLabelsMatrix());

		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65); // Use 65% of data for training

		DataSet trainingData = testAndTrain.getTrain();
		DataSet testData = testAndTrain.getTest();

		final int numInputs = 4;
		int outputNum = 3;
		long seed = 6;

		System.out.println("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.activation(Activation.TANH)
				.weightInit(WeightInit.XAVIER)
				.updater(new Sgd(0.1))
				.l2(1e-4)
				.list()
				.layer(new DenseLayer.Builder().nIn(numInputs).nOut(3)
						.build())
				.layer(new DenseLayer.Builder().nIn(3).nOut(3)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX) // Override the global TANH activation with softmax for this
														// layer
						.nIn(3).nOut(outputNum).build())
				.build();

		// run the model
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		System.out.println("model inited");
		// record score once every 100 iterations
		model.setListeners(new ScoreIterationListener(100));

		for (int i = 0; i < 1000; i++) {
			model.fit(trainingData);
		}
		System.out.println("model trained");

		// evaluate the model on the test set
		Evaluation eval = new Evaluation(3);
		INDArray output = model.output(testData.getFeatures());
		eval.eval(testData.getLabels(), output);
		System.out.println(eval.stats());

	}

	/*
	 * THIS WORKS::
	 * java -cp /Users/staffan/git/cpsign-dl4j/target/cpsign-dl4j.jar:target/cpsign
	 * com.arosbio.modeling.app.cli.CPSignApp explain scorer
	 */

	@Test
	public void testServiceLoader() {
		Iterator<MLAlgorithm> iter = FuzzyServiceLoader.iterator(MLAlgorithm.class);

		boolean containsClfier = false, containsRegressor = false;
		while (iter.hasNext()) {
			MLAlgorithm alg = iter.next();
			if (alg instanceof DLRegressor)
				containsRegressor = true;
			else if (alg instanceof DLClassifier) {
				containsClfier = true;
			}
			// System.err.println(alg.getClass().getName());
		}
		Assert.assertTrue(containsClfier);
		Assert.assertTrue(containsRegressor);
	}

	@Test
	public void getParametersThroughCPSign() throws Exception {
		CLIConsole.getInstance().setVerbosity(VerbosityLvl.SILENT);
		Assert.assertEquals(Integer.valueOf(0), new ExplainArgument.MLAlgInfo().call());
	}

	@Test
	public void testConfigStuff() throws Exception {

		DLClassifier clf = new DLClassifier();
		Map<String, Object> params = new HashMap<>();
		clf.setConfigParameters(params);

		// As a single string
		params.put("layers", Arrays.asList(10,20,30));
		clf.setConfigParameters(params);
		List<Integer> layers = clf.getHiddenLayerWidths();
		Assert.assertEquals(Arrays.asList(10, 20, 30), layers);

		// As int-array
		params.put("layers", new int[] { 30, 20, 10 });
		clf.setConfigParameters(params);
		layers = clf.getHiddenLayerWidths();
		Assert.assertEquals(Arrays.asList(30, 20, 10), layers);

		// As list of string
		params.put("layers", Arrays.asList("5", "10", "15"));
		clf.setConfigParameters(params);
		layers = clf.getHiddenLayerWidths();
		Assert.assertEquals(Arrays.asList(5, 10, 15), layers);

		// String array
		params.put("layers", new String[] { "50", "100", "015" });
		clf.setConfigParameters(params);
		layers = clf.getHiddenLayerWidths();
		Assert.assertEquals(Arrays.asList(50, 100, 15), layers);

		// Invalid input
		params.put("layers", new String[] { "a", "b", "c" });
		try {
			clf.setConfigParameters(params);
			Assert.fail();
		} catch (IllegalArgumentException e) {
		}

		// Updater
		params.put("updater", "adam;0.05");
		params.remove("layers");

		clf.setConfigParameters(params);
		IUpdater up = clf.config.getIUpdater();
		Assert.assertTrue(up instanceof Adam);
		Assert.assertEquals(.05,((Adam)up).getLearningRate(),0.0001);
		
		clf.releaseResources();

	}

	@Test
	public void testSetPathOfTrainLog() throws Exception {
		NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder()
				.activation(Activation.TANH)
				.weightInit(WeightInit.XAVIER)
				.updater(new Sgd(0.1))
				.l2(1e-4);

		DLClassifier clf = new DLClassifier(config);
		clf.numEpoch(100).testSplitFraction(0).numHiddenLayers(3).batchSize(-1).evalInterval(10)
				.lossOutput("test_out/train_out.csv");

		SubSet allData = getIrisClassificationData();
		allData.shuffle();

		allData = new Standardizer().fitAndTransform(allData);

		clf.train(allData);
		clf.releaseResources();
	}

	/*
	 * This specific split (using this rng) made train-examples with less features
	 * which set the inputWidth of the network to smaller than
	 * some of the test-examples - which made caused it to fail with
	 * indexoutofbounds in ND4JUtils conversion of the test-examples at
	 * predict-time.
	 */
	@Test
	public void testDataHasMoreFeaturesThanTrainData() throws Exception {

		GlobalConfig.getInstance().setRNGSeed(1637217339288l);
		// Predictor
		DLClassifier clf = new DLClassifier();
		clf.nEpoch(1000).inputDropOut(.3).testSplitFraction(0d); // .batchNorm(true)

		ACPClassifier acp = new ACPClassifier(new InverseProbabilityNCM(clf), new RandomSampling(1, .2));

		// Data
		ChemDataset ds = new ChemDataset(new ECFP4());
		ds.initializeDescriptors();

		ds.add(new SDFile(TestDL4JClassifier.class.getResource(AmesBinaryClf.AMES_REL_PATH).toURI()).getIterator(),
				AmesBinaryClf.PROPERTY, AmesBinaryClf.LABELS);
		
		SubSet[] splits = ds.getDataset().splitRandom(1242512,.3);
		SubSet testData = splits[0]; // 30% as test-data
		SubSet trainData = splits[1]; // 70% as train-data
		

		int maxTestFeatureIndex = DataUtils.getMaxFeatureIndex(testData);

		// Manually remove so the largest feature index is smaller than max test-features index
		trainData = new DropColumnSelector(new ColumnSpec(Range.atLeast(maxTestFeatureIndex-20)))
			.fitAndTransform(trainData);
		Assert.assertTrue("should had have at least 20 features less", DataUtils.getMaxFeatureIndex(trainData) <= (maxTestFeatureIndex-20));

		// Set the edited 70% data as train-data
		ds.withDataset(trainData); 
		// System.err.println("max index: " +
		// (DataUtils.getMaxFeatureIndex(ds.getDataset())+1));
		// System.err.println(ds);

		TestRunner tester = new TestRunner.Builder(new RandomSplit()).build();
		List<Metric> mets = tester.evaluate(ds, acp);
		Assert.assertFalse(mets.isEmpty());
		// System.err.println(mets);
	}

	/**
	 * CLI test checking that the parsing of updater works as expected
	 * 
	 * @throws Exception
	 */
	@Test
	public void testCLITuneScorer() throws Exception {
		// Remove earlier files (if any)
		String relativeOutFile = "test_out/best_params.txt";
		try {
			new File(new File("").getAbsoluteFile(), relativeOutFile).delete();
		} catch (Exception e) {
		}

		// Load the true data
		SubSet data = UnitTestBase.getIrisClassificationData();
		RobustScaler scaler = new RobustScaler();
		data = scaler.fitAndTransform(data);

		// Save the data as a fake CPSign precomputed data set
		ChemDataset ds = new ChemDataset(DescriptorFactory.getCDKDescriptorsNo3D().subList(0, 5));
		ds.withDataset(data);
		ds.initializeDescriptors();
		ds.setTextualLabels(new NamedLabels("setosa", "versicolor", "virginica"));
		File dataFile = File.createTempFile("data", ".zip");

		ModelSerializer.saveDataset(ds, new ModelInfo("iris"), dataFile,
				null);

		CPSignApp.main(new String[] { TuneScorer.CMD_NAME,
				"--data-set", dataFile.toString(),
				"--scorer", "dl-classifier:nEpoch=100:updater=Sgd;0.5:trainOutput=test_out/tune_dl_clf.csv", // this is
																												// not
																												// optimal,
																												// simply
																												// trying
																												// if
																												// things
																												// are
																												// picked
																												// up
																												// correctly
				"--test-strategy", "TestTrainSplit:stratify=True",
				"--grid", "updater=Sgd;0.01", // ,Sgd;0.1", //"width=5,10,15", //
				"-rf", "tsv",
				"--generate@file", relativeOutFile
		});

		// Test to train a model using the @file - to check if cpsign can parse it
		// correctly
		File trainedModel = File.createTempFile("trained-predictor", ".zip");
		CPSignApp.main(new String[] {
				Train.CMD_NAME,
				"--data-set", dataFile.toString(),
				// The settings produced in tune-scorer above
				"@" + relativeOutFile,
				"-mo", trainedModel.getAbsolutePath(),
		});
	}

	@Test
	public void testClonePredictor() throws InterruptedException {
		long seed = 56789;
		DLClassifier clf = new DLClassifier();
		clf.numEpoch(200) //1000
			.testSplitFraction(0)
			.numHiddenLayers(3)
			.batchSize(-1)
			.updater(new Sgd(0.1))
			.lossOutput("some_path/file.csv")
			.networkWidth(42)
			.activation(Activation.TANH)
			.setSeed(seed);
		// System.err.println(clf.getProperties());
		Thread.sleep(1000l);

		DLClassifier clone = clf.clone();
		// System.err.println(clone.getProperties());
		
//		System.err.println(clf.getProperties());
		Assert.assertEquals(Long.valueOf(seed), clf.getSeed());
		Assert.assertEquals(Long.valueOf(seed), clone.getSeed());
		
		// Verify settings are the same
		Assert.assertEquals(clf.getProperties(), clone.getProperties());
		Assert.assertTrue(clf.getProperties().containsKey("iterTimeout"));
		
		
		clf.releaseResources();
		clone.releaseResources();
	}

	@Test
	public void testSeedPickedupFromCLI(){
		
	}

	// @Rule
	// public final EnvironmentVariables env = new EnvironmentVariables();

	@Test
	public void testRepeatableResultsSameSeed()throws Exception{
		long seed = 56789;
		double s1 = trainAndEval(seed);

		// env.set("OMP_NUM_THREADS", "1");
		double s2 = trainAndEval(seed);

		// env.set("OMP_NUM_THREADS", "5");
		double s3 = trainAndEval(seed);

		// env.set("OMP_NUM_THREADS", "10");
		double s4 = trainAndEval(seed);

		// System.err.printf("%s  %s  %s  %s%n",s1,s2,s3,s4);

		Assert.assertEquals(s1, s2,0.000001);
		Assert.assertEquals(s1, s3,0.000001);
		Assert.assertEquals(s1, s4,0.000001);
	}


	private double trainAndEval(long seed)throws Exception{
		// System.err.println(System.getenv("OMP_NUM_THREADS"));
		GlobalConfig.getInstance().setRNGSeed(seed);
		// Predictor
		DLClassifier clf = new DLClassifier().clone(); 
		clf.nEpoch(1000).testSplitFraction(0d);
		// System.err.println(clf.getProperties());
		ACPClassifier acp = new ACPClassifier(new InverseProbabilityNCM(clf), new RandomSampling(1, .2));
		
		// Data
		ChemDataset ds = new ChemDataset(new ECFP4());
		ds.initializeDescriptors();
		ds.add(new SDFile(TestDL4JClassifier.class.getResource(AmesBinaryClf.AMES_REL_PATH).toURI()).getIterator(),AmesBinaryClf.PROPERTY,AmesBinaryClf.LABELS);
//		System.err.println("max index: " + (DataUtils.getMaxFeatureIndex(ds.getDataset())+1));
//		System.err.println(ds);
		RandomSplit splitter = new RandomSplit();
		splitter.setSeed(seed);
		TestRunner tester = new TestRunner.Builder(splitter).build();
		List<Metric> mets = tester.evaluate(ds, acp, ImmutableList.of(new BalancedAccuracy()));
		return ((BalancedAccuracy)mets.get(0)).getScore();

	}

}
