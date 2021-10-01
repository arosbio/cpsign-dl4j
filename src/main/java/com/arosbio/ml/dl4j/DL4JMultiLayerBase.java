package com.arosbio.ml.dl4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.arosbio.commons.CollectionUtils;
import com.arosbio.commons.TypeUtils;
import com.arosbio.commons.config.Configurable;
import com.arosbio.commons.config.IntegerConfigParameter;
import com.arosbio.modeling.CPSignSettings;

public class DL4JMultiLayerBase implements Configurable {
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DL4JMultiLayerBase.class);
//	private static final Logger LOGGER = LoggerFactory.getLogger(DL4JMultiLayerBase.class);
	public static final double DEFAULT_WEIGHT_DECAY = 0.05;
	public static final double DEFAULT_LR = 0.5;
	public static final int DEFAULT_NUM_HIDDEN_LAYERS = 5;
	public static final int DEFAULT_NETWORK_WIDTH = 10;
	public static final int DEFAULT_N_EPOCH = 10;

	// Settings
	protected long seed = CPSignSettings.getInstance().getRNGSeed();
	/** The width of the hidden layers in the network - all will have the same width */
	protected int networkWidth = DEFAULT_NETWORK_WIDTH;
	protected int numHiddenLayers = DEFAULT_NUM_HIDDEN_LAYERS;
	protected int numEpoch = DEFAULT_N_EPOCH;
	private Integer batchSize;
	protected boolean saveUpdater = false;
	protected NeuralNetConfiguration.Builder config;
	protected LossFunctions.LossFunction loss = LossFunctions.LossFunction.MCXENT;
	
	// Not required to save
	protected transient int printInterval = -1;
	protected transient DataType dType = DataType.FLOAT;


	public DL4JMultiLayerBase() {

		config = new NeuralNetConfiguration.Builder()
				.activation(Activation.RELU)
				.weightInit(WeightInit.XAVIER)
				.updater(new Sgd(DEFAULT_LR))
				.weightDecay(DEFAULT_WEIGHT_DECAY);
	}

	public DL4JMultiLayerBase(NeuralNetConfiguration.Builder config) {
		this.config = config;
	}
	
	public DL4JMultiLayerBase setNumEpoch(int nEpoch) {
		this.numEpoch = nEpoch;
		return this;
	}
	
	/**
	 * Set a fixed mini-batch size, default is to use 10 mini-batches for each epoch. If setting a negative value 
	 * there will be no mini batches and instead all data will be given for a single batch per epoch.
	 * @param batchSize
	 * @return The instance
	 */
	public DL4JMultiLayerBase setBatchSize(int batchSize) {
		this.batchSize = batchSize;
		return this;
	}
	
	public DL4JMultiLayerBase setNumHiddenLayers(int nLayers) {
		this.numHiddenLayers = nLayers;
		return this;
	}
	
	public DL4JMultiLayerBase setNetworkWidth(int width) {
		this.networkWidth = width;
		return this;
	}
	
	public DL4JMultiLayerBase setLossFunc(LossFunction loss) {
		this.loss = loss;
		return this;
	}
	
	public DL4JMultiLayerBase setLoggingInterval(int interval) {
		this.printInterval = interval;
		return this;
	}
	
	public DL4JMultiLayerBase setUpdater(IUpdater updater) {
		this.config.updater(updater);
		return this;
	}
	
	public DL4JMultiLayerBase setActivation(IActivation activation) {
		this.config.activation(activation);
		return this;
	}
	
	public DL4JMultiLayerBase setActivation(Activation activation) {
		this.config.activation(activation);
		return this;
	}
	
	public DL4JMultiLayerBase setWeightInit(WeightInit init) {
		this.config.weightInit(init);
		return this;
	}
	
	/**
	 * Set the regularization, either the weightDecay (preferred) or L1/L2 weight penalty.
	 * @param weightDecay a value equal or smaller than zero disable weightDecay regularization
	 * @param l1 a value equal or smaller than zero disable l1 regularization
	 * @param l2 a value equal or smaller than zero disable l2 regularization
	 * @return
	 */
	public DL4JMultiLayerBase setRegularization(double weightDecay, double l1, double l2) {
		if (l1>0)
			config.l1(l1);
		if (l2>0)
			config.l2(l2);
		// Disable (set to 0) if negative weight-decay is given
		config.weightDecay(Math.max(0,weightDecay));
		return this;
	}
	
	public DL4JMultiLayerBase setDType(DataType type) {
		this.dType = type;
		return this;
	}
	
	public Map<String, Object> getProperties() {
		// TODO Auto-generated method stub
		return new HashMap<>();
	}

	public void setSeed(long seed) {
		this.seed = seed;
	}

	public long getSeed() {
		return this.seed;
	}

	public int calcBatchSize(int numTrainingSamples) {
		if (batchSize == null) {
			return Math.max(10, numTrainingSamples/10);
		} else if (batchSize < 0) {
			// Use all examples each epoch (a single batch)
			return numTrainingSamples;
		} else {
			return batchSize;
		}
		
	}
	private static List<String> NET_WIDTH_CONF_NAMES = Arrays.asList("width","networkWidth","hiddenLayerWidth");
	private static List<String> N_HIDDEN_CONF_NAMES = Arrays.asList("depth","numHidden","numHiddenLayers");
	private static List<String> N_EPOCH_CONF_NAMES = Arrays.asList("nEpoch", "numEpoch");
	private static List<String> BATCH_SIZE_CONF_NAMES = Arrays.asList("batchSize", "miniBatch");
	
	@Override
	public List<ConfigParameter> getConfigParameters() {
		List<ConfigParameter> confs = new ArrayList<>();
		confs.add(new IntegerConfigParameter(N_HIDDEN_CONF_NAMES, DEFAULT_NUM_HIDDEN_LAYERS).addDescription("Number of hidden layers"));
		confs.add(new IntegerConfigParameter(NET_WIDTH_CONF_NAMES, DEFAULT_NETWORK_WIDTH).addDescription("The width of each hidden layer, the input and output-layers are defined by the required input and output"));
		confs.add(new IntegerConfigParameter(N_EPOCH_CONF_NAMES, DEFAULT_N_EPOCH).addDescription("The number of epochs"));
		confs.add(new IntegerConfigParameter(BATCH_SIZE_CONF_NAMES, null)
				.addDescription("The mini-batch size, will control how many records are passed in each iteration. The default is to use 10 mini-batches for each epoch "
						+ "- thus calculated at training-time. Setting a value smaller than 0 disables mini-batches and passes _all data_ through at the same time (i.e., one batch per epoch)."));
		return confs;
	}

	@Override
	public void setConfigParameters(Map<String, Object> params) 
			throws IllegalStateException, IllegalArgumentException {

		for (Map.Entry<String, Object> c : params.entrySet()) {
			String key = c.getKey();
			if (CollectionUtils.containsIgnoreCase(NET_WIDTH_CONF_NAMES, key)) {
				networkWidth = TypeUtils.asInt(c.getValue());
			} else if (CollectionUtils.containsIgnoreCase(N_HIDDEN_CONF_NAMES, key)) {
				numHiddenLayers = TypeUtils.asInt(c.getValue());
			} else if (CollectionUtils.containsIgnoreCase(N_EPOCH_CONF_NAMES, key)) {
				numEpoch = TypeUtils.asInt(c.getValue());
			} else if (CollectionUtils.containsIgnoreCase(BATCH_SIZE_CONF_NAMES, key)) {
				if (c.getValue() == null)
					batchSize = null;
				else
					batchSize = TypeUtils.asInt(c.getValue());
			} 
//			else if (CollectionUtils.containsIgnoreCase(N_HIDDEN_CONF_NAMES, key)) {
//				
//			} 
			else {
				LOGGER.debug("Unused Config-argument: " + c);
			}
		}
	}

	

//	@Override
//	public DL4JMultiLayerBase clone() {
//		
//		// TODO Auto-generated method stub
//		return null;
//	}

	
	protected void copyParametersToNew(DL4JMultiLayerBase cpy) {
		cpy.config = config.clone();
		cpy.seed = seed;
		cpy.networkWidth = networkWidth;
		cpy.numHiddenLayers = numHiddenLayers;
		cpy.numEpoch = numEpoch;
		cpy.batchSize = batchSize;
		cpy.saveUpdater = saveUpdater;
		cpy.loss = loss;
		cpy.printInterval = printInterval;
		cpy.dType = dType;
	}
	

}
