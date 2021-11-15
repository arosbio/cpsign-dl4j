package com.arosbio.ml.dl4j;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.arosbio.ml.nd4j.ND4JUtil;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.data.DataUtils;
import com.arosbio.modeling.data.FeatureVector;
import com.arosbio.modeling.ml.algorithms.Regressor;

public class DLRegressor extends DL4JMultiLayerBase implements Regressor {

	public static final String NAME = "DLRegressor";
	public static final String DESCRIPTION = "A Deeplearning/Artifical Neural Network (DL/ANN) for regression, implemented in Deeplearning4J";
	public static final int ID = 4;
	public static final LossFunction DEFAULT_LOSS_FUNC = LossFunction.MSE;
	
	public DLRegressor() {
		super();
		setLossFunc(DEFAULT_LOSS_FUNC);
	}

	public DLRegressor(NeuralNetConfiguration.Builder config) {
		super(config);
		setLossFunc(DEFAULT_LOSS_FUNC);
	}
	
	@Override
	public Map<String, Object> getProperties() {
		// TODO
		return super.getProperties();
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
	public void train(List<DataRecord> trainingset) throws IllegalArgumentException {
		if (numHiddenLayers < 1)
			throw new IllegalStateException("Number of network layers must be at least 1");

		inputWidth = DataUtils.getMaxFeatureIndex(trainingset)+1;
		
		// Create the list builder and add the input layer
		ListBuilder listBldr = config.seed(seed).list();
		
		int lastW = addHiddenLayers(listBldr, inputWidth);

		// Add output layer
		listBldr.layer( new OutputLayer.Builder(loss)
				.activation(Activation.IDENTITY) // Override the global activation
				.nIn(lastW).nOut(1).build());
		
		trainNetwork(listBldr.build(), trainingset, false);
	}
	
	@Override
	public double predictValue(FeatureVector example) throws IllegalStateException {
		INDArray pred = model.output(ND4JUtil.toArray(example, inputWidth));
		return pred.getDouble(0,0);
	}

	@Override
	public DLRegressor clone() {
		DLRegressor clone = new DLRegressor(null);
		super.copyParametersToNew(clone);
		return clone;
	}


}
