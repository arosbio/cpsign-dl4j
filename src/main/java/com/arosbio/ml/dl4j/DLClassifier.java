package com.arosbio.ml.dl4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.arosbio.ml.nd4j.ND4JUtil;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.data.DataUtils;
import com.arosbio.modeling.data.FeatureVector;
import com.arosbio.modeling.ml.algorithms.MultiLabelClassifier;
import com.arosbio.modeling.ml.algorithms.ScoringClassifier;

public class DLClassifier extends DL4JMultiLayerBase 
	implements ScoringClassifier, MultiLabelClassifier {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DLClassifier.class);

	public static final String NAME = "DLClassifier";
	public static final String DESCRIPTION = "A Deeplearning/Artifical Neural Network (DL/ANN) for classification, implemented in Deeplearning4J";
	public static final int ID = 17;
	public static final LossFunction DEFAULT_LOSS_FUNC = LossFunction.MCXENT;

	private transient int[] labels;

	public DLClassifier() {
		super();
		setLossFunc(DEFAULT_LOSS_FUNC);
	}

	public DLClassifier(NeuralNetConfiguration.Builder config) {
		super(config);
		setLossFunc(DEFAULT_LOSS_FUNC);
	}
	
	@Override
	public Map<String, Object> getProperties() {
		return super.getProperties();
	}

	@Override
	public void setSeed(long seed) {
		this.seed = seed;
	}

	@Override
	public long getSeed() {
		return this.seed;
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
	public void setConfigParameters(Map<String, Object> params) 
			throws IllegalStateException, IllegalArgumentException {
		super.setConfigParameters(params);
	}

	@Override
	public List<Integer> getLabels() {
		List<Integer> labels = new ArrayList<>();
		if (model !=null) {
			for (int l : model.getLabels().toIntVector()) {
				labels.add(l);
			}
			return labels;
		}
		return labels;
	}

	@Override
	public int predictClass(FeatureVector feature) throws IllegalStateException {
		if (model==null)
			throw new IllegalStateException("Model not trained yet");
		return model.predict(ND4JUtil.toArray(feature, inputWidth))[0];
	}
	
	@Override
	public Map<Integer, Double> predictScores(FeatureVector feature) 
			throws IllegalStateException {
		if (model==null)
			throw new IllegalStateException("Model not trained yet");

		INDArray pred = model.output(ND4JUtil.toArray(feature, inputWidth));
		if (labels == null) {
			// use an int-array instead for quick lookup and cache the labels
			labels = model.getLabels().toIntVector();
		}
		Map<Integer,Double> predMapping = new HashMap<>();
		for (int i=0; i<labels.length; i++) {
			predMapping.put(labels[i], pred.getDouble(0,i));
		}
		return predMapping;
	}
	

	@Override
	public void train(List<DataRecord> trainingset) throws IllegalArgumentException {
		
		inputWidth = DataUtils.getMaxFeatureIndex(trainingset)+1;
		int numOutputs = DataUtils.countLabels(trainingset).size();

		// Create the list builder and add the input layer
		ListBuilder listBldr = config.seed(seed).dataType(dType).list();

		// Add hidden layers
		int lastW = addHiddenLayers(listBldr,inputWidth);
		
		// Add output layer
		listBldr.layer( new OutputLayer.Builder(loss) 
				.activation(Activation.SOFTMAX)
				.nIn(lastW).nOut(numOutputs)
				.dropOut(1) // no drop out for the output layer
				.build()
				);

		trainNetwork(listBldr.build(), trainingset, true);		
	}
	
	@Override
	public DLClassifier clone() {
		DLClassifier clone = new DLClassifier(null);
		super.copyParametersToNew(clone);
		return clone;
	}
	
}
