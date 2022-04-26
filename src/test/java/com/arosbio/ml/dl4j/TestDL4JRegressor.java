package com.arosbio.ml.dl4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import com.arosbio.commons.logging.LoggerUtils;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.data.Dataset;
import com.arosbio.modeling.data.Dataset.SubSet;
import com.arosbio.modeling.data.transform.scale.RobustScaler;
import com.arosbio.modeling.data.transform.scale.Standardizer;
import com.arosbio.modeling.ml.metrics.MetricFactory;
import com.arosbio.modeling.ml.metrics.SingleValuedMetric;
import com.arosbio.modeling.ml.metrics.regression.MAE;
import com.arosbio.modeling.ml.metrics.regression.R2;
import com.arosbio.modeling.ml.metrics.regression.RMSE;
import com.arosbio.modeling.ml.testing.RandomSplit;
import com.arosbio.modeling.ml.testing.TestRunner;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.learning.config.Sgd;

import test_utils.UnitTestBase;

public class TestDL4JRegressor extends UnitTestBase {

	@Test
	public void testTrainPredict() throws Exception {
		SubSet trainingData = getAndrogenReceptorRegressionData();

		DLRegressor model = new DLRegressor();
		model
			.numEpoch(2000)
			.networkWidth(10)
			.numHiddenLayers(5)
			.updater(new Sgd(0.05))
			.evalInterval(1);

		
		Standardizer std = new Standardizer();
		trainingData = std.fitAndTransform(trainingData);

		List<DataRecord> testRecs = new ArrayList<>(trainingData.subList(0, 100));
		trainingData.subList(0, 100).clear();

		LoggerUtils.setDebugMode(System.out);
		model.train(trainingData);

		MAE absErr = new MAE();
		R2 r2 = new R2();
		RMSE rmse = new RMSE();

		for (DataRecord r : testRecs) {
			double y_hat = model.predictValue(r.getFeatures());
			absErr.addPrediction(r.getLabel(), y_hat);
			r2.addPrediction(r.getLabel(), y_hat);
			rmse.addPrediction(r.getLabel(), y_hat);
		}
		

		System.out.println(absErr);
		System.out.println(r2);
		System.out.println(rmse);



		// Save / Re-load model from file to check same predictions are generated
		// Save the model
		File modelFile = File.createTempFile("model", ".net"); 
		try (OutputStream ostream = new FileOutputStream(modelFile)){
			model.saveToStream(ostream);
		}

		// Load it from file
		DLRegressor loaded = new DLRegressor();
		try (InputStream istream = new FileInputStream(modelFile);){
			loaded.loadFromStream(istream);
		}


		MAE absErr2 = new MAE();
		R2 r2_2 = new R2();
		RMSE rmse2 = new RMSE();

		for (DataRecord r : testRecs) {
			double y_hat = loaded.predictValue(r.getFeatures());
			absErr2.addPrediction(r.getLabel(), y_hat);
			r2_2.addPrediction(r.getLabel(), y_hat);
			rmse2.addPrediction(r.getLabel(), y_hat);
		}

		System.out.println(absErr2);
		System.out.println(r2_2);
		System.out.println(rmse2);
		
		Assert.assertEquals(absErr.getScore(), absErr2.getScore(), .00001);
		Assert.assertEquals(r2.getScore(), r2_2.getScore(), .00001);
		Assert.assertEquals(rmse.getScore(), rmse2.getScore(), .00001);

		LoggerUtils.reloadLogger();
		
	}
	
	@Test
	public void testDropout() throws IOException {
		DLRegressor regressor = new DLRegressor();
		regressor.updater(new Sgd(0.05)).dropOut(.2).inputDropOut(.1).batchSize(48).nEpoch(300);
		
		SubSet data = getAndrogenReceptorRegressionData();
		
		RobustScaler std = new RobustScaler();
		data = std.fitAndTransform(data);
		
		TestRunner runner = new TestRunner.Builder(new RandomSplit(.2)).build();
		Dataset ds = new Dataset();
		ds.setDataset(data);
		List<SingleValuedMetric> metrics = MetricFactory.filterToSingleValuedMetrics(MetricFactory.getRegressorMetrics());
		runner.evaluateRegressor(ds, regressor, metrics);
		System.err.println(metrics);
//		[MAE : 0.6156956466600819, R^2 : 0.37198641176608016, RMSE : 0.7879424842476385]
		regressor.releaseResources();
	}
	
	@Test
	public void testWeightDecay() throws IOException {
		DLRegressor regressor = new DLRegressor();
		regressor.updater(new Sgd(0.05)).weightDecay(.05).batchSize(48).nEpoch(300);
		
		SubSet data = getAndrogenReceptorRegressionData();
		
		RobustScaler std = new RobustScaler();
		data = std.fitAndTransform(data);
		
		TestRunner runner = new TestRunner.Builder(new RandomSplit(.2)).build();
		Dataset ds = new Dataset();
		ds.setDataset(data);
		List<SingleValuedMetric> metrics = MetricFactory.filterToSingleValuedMetrics(MetricFactory.getRegressorMetrics());
		runner.evaluateRegressor(ds, regressor, metrics);
		System.err.println(metrics);
//		[MAE : 0.5570755737478232, R^2 : 0.46266618968589535, RMSE : 0.7607753285278875]
		regressor.releaseResources();
	}
	
	@Test
	public void checkDefaultOptAlg() {
		System.err.println(new NeuralNetConfiguration.Builder().getIUpdater());
	}
	
}
