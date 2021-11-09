package com.arosbio.ml.dl4j;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.arosbio.ml.nd4j.ND4JUtil;
import com.arosbio.ml.nd4j.ND4JUtil.DataConverter;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.data.DataUtils;
import com.arosbio.modeling.data.FeatureVector;
import com.arosbio.modeling.ml.algorithms.Regressor;

public class DL4JMultiLayerRegressor extends DL4JMultiLayerBase implements Regressor {

	private static final Logger LOGGER = LoggerFactory.getLogger(DL4JMultiLayerRegressor.class);

	public static final String NAME = "DL4JMultiLayerRegressor";
	public static final String DESCRIPTION = "A Deeplearning/Artifical Neural Network (DL/ANN) for regression, implemented in Deeplearning4J";
	public static final int ID = 4;
	public static final LossFunction DEFAULT_LOSS_FUNC = LossFunction.MSE;

	// Settings

	// Not required to save
	private transient int inputWidth;
	
	// Only != null when model has been trained
	private MultiLayerNetwork model;
	

	public DL4JMultiLayerRegressor() {
		super();
		setLossFunc(DEFAULT_LOSS_FUNC);
	}

	public DL4JMultiLayerRegressor(NeuralNetConfiguration.Builder config) {
		super(config);
		setLossFunc(DEFAULT_LOSS_FUNC);
	}
	
	@Override
	public Map<String, Object> getProperties() {
		// TODO
		return super.getProperties();
	}

	@Override
	public boolean isFitted() {
		return model != null;
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
	public List<ConfigParameter> getConfigParameters() {
		return super.getConfigParameters();
	}

	@Override
	public void setConfigParameters(Map<String, Object> params) throws IllegalStateException, IllegalArgumentException {
		super.setConfigParameters(params);
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
		
		// Create the network
		model = new MultiLayerNetwork(listBldr.build());
		model.init();
		model.setListeners(new EpochScoreListener());

		// calculate batch size
		int batch = calcBatchSize(trainingset.size());
		
		DataConverter conveter = DataConverter.regression(trainingset);
		DataSetIterator iter = new INDArrayDataSetIterator(conveter, batch);

		model.fit(iter, numEpoch);
	}
	
	@Override
	public double predictValue(FeatureVector example) throws IllegalStateException {
		INDArray pred = model.output(ND4JUtil.toArray(example, inputWidth));
		return pred.getDouble(0,0);
	}

	@Override
	public void saveToStream(OutputStream ostream) throws IOException, IllegalStateException {
		if (model==null)
			throw new IllegalStateException("Model not trained yet");
		LOGGER.debug("Saving {} model to stream",NAME);
		ModelSerializer.writeModel(model, ostream, saveUpdater);
	}

	@Override
	public void loadFromStream(InputStream istream) throws IOException {
		LOGGER.debug("Attempting to load {} model", NAME);
		model = ModelSerializer.restoreMultiLayerNetwork(istream);
		inputWidth = model.getLayer(0).getParam("W").rows();
		LOGGER.debug("Finished loading DL4J model with properties: " + model.summary());
	}



	@Override
	public DL4JMultiLayerRegressor clone() {
		DL4JMultiLayerRegressor clone = new DL4JMultiLayerRegressor(null);
		super.copyParametersToNew(clone);
		return clone;
	}


}
