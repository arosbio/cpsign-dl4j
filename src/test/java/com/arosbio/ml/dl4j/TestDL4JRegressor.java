package com.arosbio.ml.dl4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.learning.config.Sgd;

import com.arosbio.commons.logging.LoggerUtils;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.data.Dataset.SubSet;
import com.arosbio.modeling.data.transform.scale.Standardizer;
import com.arosbio.modeling.ml.metrics.regression.MAE;
import com.arosbio.modeling.ml.metrics.regression.R2;
import com.arosbio.modeling.ml.metrics.regression.RMSE;

import test_utils.UnitTestBase;

public class TestDL4JRegressor extends UnitTestBase {

	@Test
	public void testTrainPredict() throws Exception {
		SubSet trainingData = getAndrogenReceptorRegressionData();

		DLRegressor model = new DLRegressor();
		model
			.setNumEpoch(2000)
			.setNetworkWidth(10)
			.setNumHiddenLayers(5)
			.setUpdater(new Sgd(0.05))
			.setLoggingInterval(1);

		
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
	
}
