package com.arosbio.ml.dl4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

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
import com.arosbio.commons.FuzzyServiceLoader;
import com.arosbio.commons.logging.LoggerUtils;
import com.arosbio.ml.nd4j.ND4JUtil.DataConverter;
import com.arosbio.modeling.CPSignSettings;
import com.arosbio.modeling.app.cli.CPSignApp;
import com.arosbio.modeling.app.cli.ExplainArgument;
import com.arosbio.modeling.app.cli.TuneScorer;
import com.arosbio.modeling.cheminf.ChemDataset;
import com.arosbio.modeling.cheminf.NamedLabels;
import com.arosbio.modeling.cheminf.descriptors.DescriptorFactory;
import com.arosbio.modeling.cheminf.descriptors.fp.ECFP4;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.data.Dataset.SubSet;
import com.arosbio.modeling.data.transform.scale.RobustScaler;
import com.arosbio.modeling.data.transform.scale.Standardizer;
import com.arosbio.modeling.io.ModelCreator;
import com.arosbio.modeling.io.ModelInfo;
import com.arosbio.modeling.io.PrecomputedDataClassification;
import com.arosbio.modeling.ml.algorithms.MLAlgorithm;
import com.arosbio.modeling.ml.cp.acp.ACPClassifier;
import com.arosbio.modeling.ml.cp.nonconf.classification.InverseProbabilityNCM;
import com.arosbio.modeling.ml.ds_splitting.RandomSampling;
import com.arosbio.modeling.ml.metrics.Metric;
import com.arosbio.modeling.ml.metrics.classification.BalancedAccuracy;
import com.arosbio.modeling.ml.metrics.classification.ClassifierAccuracy;
import com.arosbio.modeling.ml.testing.RandomSplit;
import com.arosbio.modeling.ml.testing.TestRunner;

import test_utils.UnitTestBase;

public class TestDL4JClassifier extends UnitTestBase {

	@Test
	public void testTrainSaveAndLoad() throws IllegalArgumentException, IOException {
		
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
		
		// Train it
		clf.train(trainingData);

		// Evaluate the first one
		BalancedAccuracy ba = new BalancedAccuracy();
		ClassifierAccuracy ca = new ClassifierAccuracy();
		
		
		for (DataRecord r : testData) {
			int pred = clf.predictClass(r.getFeatures());
			ba.addPrediction((int)r.getLabel(),pred);
			ca.addPrediction((int)r.getLabel(),pred);
		}
		
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
		
		BalancedAccuracy ba2 = new BalancedAccuracy();
		ClassifierAccuracy ca2 = new ClassifierAccuracy();
		
		for (DataRecord r : testData) {
			int pred = loaded.predictClass(r.getFeatures());
			ba2.addPrediction((int)r.getLabel(),pred);
			ca2.addPrediction((int)r.getLabel(),pred);
		}
		// Check that the loaded model produces the same results
		Assert.assertEquals(ba.getScore(), ba2.getScore(), 0.000001);
		Assert.assertEquals(ca.getScore(), ca2.getScore(), 0.000001);
		
		clf.close();
		loaded.close();
		
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
		
		LoggerUtils.setDebugMode(System.out);
		clf.train(allData);
		
		
		System.err.println("test label: "+testRec.getLabel());
		System.err.println("Scores: "+ clf.predictScores(allData.get(0).getFeatures()));
		clf.close();
	}
	
//	@Test
	public void testClassifyIris() throws Exception {
		SubSet data = getIrisClassificationData();
		data.shuffle();
		data = new Standardizer().fitAndTransform(data);
		
		DataConverter conv = DataConverter.classification(data);
		
		DataSet allData = new DataSet(conv.getFeaturesMatrix(), conv.getLabelsMatrix());
		
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

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
            .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                .nIn(3).nOut(outputNum).build())
            .build();
        

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println("model inited");
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        for(int i=0; i<1000; i++ ) {
            model.fit(trainingData);
        }
        System.out.println("model trained");

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());
		
	}
	
	/*
	 THIS WORKS:: 
	 java -cp /Users/staffan/git/cpsign-dl4j/target/cpsign-dl4j.jar:target/cpsign com.arosbio.modeling.app.cli.CPSignApp explain scorer
	 */
	
	@Test
	public void testServiceLoader() {
		Iterator<MLAlgorithm> iter = FuzzyServiceLoader.iterator(MLAlgorithm.class);
		
		boolean containsClfier=false, containsRegressor=false;
		while (iter.hasNext()) {
			MLAlgorithm alg = iter.next();
			if (alg instanceof DLRegressor)
				containsRegressor = true;
			else if (alg instanceof DLClassifier){
				containsClfier = true;
			}
//			System.err.println(alg.getClass().getName());
		}
		Assert.assertTrue(containsClfier);
		Assert.assertTrue(containsRegressor);
	}
	
	@Test
	public void getParametersThroughCPSign() throws Exception {
		new ExplainArgument.MLAlgInfo().call();
	}
	
	@Test
	public void testConfigStuff() throws Exception {
		
		DLClassifier clf = new DLClassifier();
		Map<String,Object> params = new HashMap<>();
		clf.setConfigParameters(params);
		
		// As a single string
		params.put("layers", "10,20,30");
		clf.setConfigParameters(params);
		List<Integer> layers = clf.getHiddenLayerWidths();
		Assert.assertEquals(Arrays.asList(10,20,30), layers);
		
		// As int-array
		params.put("layers", new int[] {30,20,10});
		clf.setConfigParameters(params);
		layers = clf.getHiddenLayerWidths();
		Assert.assertEquals(Arrays.asList(30,20,10), layers);
		
		// As list of string
		params.put("layers", Arrays.asList("5","10","15"));
		clf.setConfigParameters(params);
		layers = clf.getHiddenLayerWidths();
		Assert.assertEquals(Arrays.asList(5,10,15), layers);
		
		// String array
		params.put("layers", new String[] {"50","100","015"});
		clf.setConfigParameters(params);
		layers = clf.getHiddenLayerWidths();
		Assert.assertEquals(Arrays.asList(50,100,15), layers);
		
		// Invalid input
		params.put("layers", new String[] {"a","b","c"});
		try {
			clf.setConfigParameters(params);
			Assert.fail();
		} catch(IllegalArgumentException e) {}
		
		
		// Updater
		params.put("updater", "adam;0.05");
		params.remove("layers");
		
		clf.setConfigParameters(params);
		IUpdater up = clf.config.getIUpdater();
		Assert.assertTrue(up instanceof Adam);
		Assert.assertEquals(.05,((Adam)up).getLearningRate(),0.0001);
		
		
		clf.close();
	}
	
	@Test
	public void testSetPathOfTrainLog() throws Exception {
		NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder()
				.activation(Activation.TANH)
				.weightInit(WeightInit.XAVIER)
				.updater(new Sgd(0.1))
				.l2(1e-4);
		
		DLClassifier clf = new DLClassifier(config);
		clf.numEpoch(100).testSplitFraction(0).numHiddenLayers(3).batchSize(-1).evalInterval(10).lossOutput("test_out/train_out.csv");
		
		SubSet allData = getIrisClassificationData();
		allData.shuffle();
		
		allData = new Standardizer().fitAndTransform(allData);
		
		clf.train(allData);
		clf.close();
	}
	
	/*
	 * This specific split (using this rng) made train-examples with less features which set the inputWidth of the network to smaller than
	 * some of the test-examples - which made caused it to fail with indexoutofbounds in ND4JUtils conversion of the test-examples at predict-time.
	 */
	@Test
	public void bugSearching() throws Exception {
		
		CPSignSettings.getInstance().setRNGSeed(1637217339288l);
		// Predictor
		DLClassifier clf = new DLClassifier(); 
		clf.nEpoch(1000).testSplitFraction(0d);
		ACPClassifier acp = new ACPClassifier(new InverseProbabilityNCM(clf), new RandomSampling(1, .2));
		
		// Data
		ChemDataset ds = new ChemDataset(new ECFP4());
		ds.initializeDescriptors();
		ds.add(new SDFile(TestDL4JClassifier.class.getResource(AmesBinaryClf.AMES_REL_PATH).toURI()).getIterator(),AmesBinaryClf.PROPERTY,AmesBinaryClf.LABELS);
//		System.err.println("max index: " + (DataUtils.getMaxFeatureIndex(ds.getDataset())+1));
//		System.err.println(ds);
		
		TestRunner tester = new TestRunner(new RandomSplit());
		List<Metric> mets = tester.evaluate(ds, acp);
		Assert.assertFalse(mets.isEmpty());
//		System.err.println(mets);
	}
	
	/**
	 * CLI test checking that the parsing of updater works as expected
	 * @throws Exception
	 */
	@Test
	public void testCLITuneScorer() throws Exception {
		// Load the true data
		SubSet data = UnitTestBase.getIrisClassificationData();
		RobustScaler scaler = new RobustScaler();
		data = scaler.fitAndTransform(data);
		
		// Save the data as a fake CPSign precomputed data set
		ChemDataset ds = new ChemDataset(DescriptorFactory.getCDKDescriptorsNo3D().subList(0, 5));
		ds.setDataset(data);
		ds.initializeDescriptors();
		ds.setTextualLabels(new NamedLabels("setosa", "versicolor", "virginica"));
		File dataFile = File.createTempFile("data", ".zip");
		ModelCreator.generatePrecomputedModel(new PrecomputedDataClassification(ds, new ModelInfo("iris")), dataFile, null);
		
		CPSignApp.main(new String[] {TuneScorer.CMD_NAME, 
			"--data-set",dataFile.toString(),
			"--scorer", "dl-classifier:nEpoch=100:updater=Sgd;0.5:trainOutput=test_out/tune_dl_clf.csv", // this is not optimal, simply trying if things are picked up correctly
			"--license",UnitTestBase.getFirstLicenseFile().toString(),
			"--test-strategy", "TestTrainSplit",
			"--grid", "updater=Sgd;0.01,Sgd;0.1", //"width=5,10,15", //
			"-rf", "tsv"
		});
	}
	
	@Test
	public void testClonePredictor() {
		DLClassifier clf = new DLClassifier();
		clf.numEpoch(200) //1000
			.testSplitFraction(0)
			.numHiddenLayers(3)
			.batchSize(-1)
			.updater(new Sgd(0.1))
			.lossOutput("some_path/file.csv")
			.networkWidth(42)
			.activation(Activation.TANH);
		
		DLClassifier clone = clf.clone();
		
		// Verify settings are the same
		Assert.assertEquals(clf.getProperties(), clone.getProperties());
		
		
		clf.close();
		clone.close();
	}

}
