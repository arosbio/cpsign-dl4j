/*
 * Copyright (C) Aros Bio AB.
 *
 * CPSign is an Open Source Software that is dual licensed to allow you to choose a license that best suits your requirements:
 *
 * 1) GPLv3 (GNU General Public License Version 3) with Additional Terms, including an attribution clause as well as a limitation to use the software for commercial purposes.
 *
 * 2) CPSign Proprietary License that allows you to use CPSign for commercial activities, such as in a revenue-generating operation or environment, or integrate CPSign in your proprietary software without worrying about disclosing the source code of your proprietary software, which is required if you choose to use the software under GPLv3 license. See arosbio.com/cpsign/commercial-license for details.
 */
package com.arosbio.ml.dl4j;

import java.util.List;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.arosbio.ml.nd4j.ND4JUtil;
import com.arosbio.data.DataRecord;
import com.arosbio.data.DataUtils;
import com.arosbio.data.FeatureVector;
import com.arosbio.ml.algorithms.Regressor;

public class DLRegressor extends DL4JMultiLayerBase implements Regressor {

	public static final String NAME = "DLRegressor";
	public static final String DESCRIPTION = "A Deeplearning/Artifical Neural Network (DL/ANN) for regression, implemented in Deeplearning4J";
	public static final int ID = 4;
	public static final LossFunction DEFAULT_LOSS_FUNC = LossFunction.MSE;
	
	public DLRegressor() {
		super(DEFAULT_LOSS_FUNC);
	}

	public DLRegressor(NeuralNetConfiguration.Builder config) {
		super(DEFAULT_LOSS_FUNC,config);
	}
	
	@Override
	public String getDescription() {
		return DESCRIPTION;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public int getID() {
		return ID;
	}

	@Override
	public void fit(List<DataRecord> trainingset) throws IllegalArgumentException {
		train(trainingset);
	}

	@Override
	public void train(List<DataRecord> trainingset) throws IllegalArgumentException {

		int nIn = DataUtils.getMaxFeatureIndex(trainingset)+1;
		
		// Create the list builder and add the input layer
		ListBuilder listBldr = config.seed(getSeed()).dataType(dType).list();
		
		int lastW = addHiddenLayers(listBldr, nIn);

		// Add output layer
		listBldr.layer( new OutputLayer.Builder(loss)
				.activation(Activation.IDENTITY) // Override the global activation
				.nIn(lastW).nOut(1).build());
		
		trainNetwork(listBldr.build(), trainingset, false);
	}
	
	@Override
	public double predictValue(FeatureVector example) throws IllegalStateException {
		INDArray pred = model.output(ND4JUtil.toArray(example, getInputWidth()));
		return pred.getDouble(0,0);
	}

	@Override
	public DLRegressor clone() {
		DLRegressor clone = new DLRegressor(null);
		super.copyParametersToNew(clone);
		return clone;
	}


}
