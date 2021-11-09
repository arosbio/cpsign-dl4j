package com.arosbio.ml.dl4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
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
import com.arosbio.commons.config.EnumConfigParameter;
import com.arosbio.commons.config.IntegerConfigParameter;
import com.arosbio.commons.config.NumericConfigParameter;
import com.arosbio.modeling.CPSignSettings;

public abstract class DL4JMultiLayerBase implements Configurable {
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DL4JMultiLayerBase.class);

	public static final double DEFAULT_WEIGHT_DECAY = 0.05; // TODO
	public static final double DEFAULT_LR = 0.5;
	public static final int DEFAULT_NUM_HIDDEN_LAYERS = 5;
	public static final int DEFAULT_NETWORK_WIDTH = 10;
	public static final int DEFAULT_N_EPOCH = 10;

	// Settings
	protected long seed = CPSignSettings.getInstance().getRNGSeed();
	/** The width of the hidden layers in the network - all will have the same width */
	protected int networkWidth = DEFAULT_NETWORK_WIDTH;
	protected int numHiddenLayers = DEFAULT_NUM_HIDDEN_LAYERS;
	protected List<Integer> explicitLayerWidths = null;
	
	protected int numEpoch = DEFAULT_N_EPOCH;
	private Integer batchSize;
	private boolean batchNorm = false;
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
	
	/**
	 * Set explicit widths for each hidden layer (i.e. allows different widths for each layer). 
	 * The first index of the array is for the first layer and so on.. If <code>null</code> is sent, the
	 * {@link #setNetworkWidth(int)} and {@link #setNumHiddenLayers(int)} parameters will be used instead.
	 * @param widths An array of hidden layer widths
	 * @return The calling instance (fluid API)
	 */
	public DL4JMultiLayerBase setHiddenLayers(int... widths) {
		if (widths == null || widths.length == 0) {
			explicitLayerWidths = null;
		} else {
			explicitLayerWidths = new ArrayList<>();
			for (int w : widths) {
				explicitLayerWidths.add(w);
			}
		}
		return this;
	}
	
	/**
	 * Set explicit widths for each hidden layer (i.e. allows different widths for each layer). 
	 * The first index of the list is for the first layer and so on.. If <code>null</code> is sent, the
	 * {@link #setNetworkWidth(int)} and {@link #setNumHiddenLayers(int)} parameters will be used instead.
	 * @param widths A list of hidden layer widths, or <code>null</code> 
	 * @return The calling instance (fluid API)
	 */
	public DL4JMultiLayerBase setHiddenLayers(List<Integer> widths) {
		if (widths == null || widths.isEmpty()) {
			explicitLayerWidths = null;
			return this;
		}
		// Use a copy
		this.explicitLayerWidths = new ArrayList<>(widths);
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
	
	public DL4JMultiLayerBase setBatchNorm(boolean useBatchNorm) {
		this.batchNorm = useBatchNorm;
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
	private static List<String> WEIGHT_DECAY_CONF_NAMES = Arrays.asList("weightDecay");
	private static List<String> L2_CONF_NAMES = Arrays.asList("l2");
	private static List<String> L1_CONF_NAMES = Arrays.asList("l1");
	private static List<String> LOSS_FUNC_CONF_NAMES = Arrays.asList("loss");
	private static List<String> ACTIVATION_CONF_NAMES = Arrays.asList("activation");
	
	@Override
	public List<ConfigParameter> getConfigParameters() {
		List<ConfigParameter> confs = new ArrayList<>();
		confs.add(new IntegerConfigParameter(N_HIDDEN_CONF_NAMES, DEFAULT_NUM_HIDDEN_LAYERS).addDescription("Number of hidden layers"));
		confs.add(new IntegerConfigParameter(NET_WIDTH_CONF_NAMES, DEFAULT_NETWORK_WIDTH).addDescription("The width of each hidden layer, the input and output-layers are defined by the required input and output"));
		confs.add(new IntegerConfigParameter(N_EPOCH_CONF_NAMES, DEFAULT_N_EPOCH).addDescription("The number of epochs"));
		confs.add(new IntegerConfigParameter(BATCH_SIZE_CONF_NAMES, null)
				.addDescription("The mini-batch size, will control how many records are passed in each iteration. The default is to use 10 mini-batches for each epoch "
						+ "- thus calculated at training-time. Setting a value smaller than 0 disables mini-batches and passes _all data_ through at the same time (i.e., one batch per epoch)."));
		confs.add(new NumericConfigParameter(WEIGHT_DECAY_CONF_NAMES, DEFAULT_WEIGHT_DECAY).addDescription("The weight-decay regularization term, put to <=0 if not to use weight-decay regularization"));
		confs.add(new NumericConfigParameter(L2_CONF_NAMES, 0).addDescription("L2 regularization term. Disabled by default."));
		confs.add(new NumericConfigParameter(L1_CONF_NAMES, 0).addDescription("L1 regularization term. Disabled by default."));
		confs.add(new EnumConfigParameter<>(LOSS_FUNC_CONF_NAMES, EnumSet.allOf(LossFunction.class),loss).addDescription("The loss function of the network"));
		confs.add(new EnumConfigParameter<>(ACTIVATION_CONF_NAMES, EnumSet.allOf(Activation.class),Activation.RELU).addDescription("The activation function for the hidden layers"));
		
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
			} else if (CollectionUtils.containsIgnoreCase(WEIGHT_DECAY_CONF_NAMES, key)) {
				double decay = TypeUtils.asDouble(c.getValue());
				if (decay <= 0)
					config.weightDecay(0);
				else 
					config.weightDecay(decay);
			} else if (CollectionUtils.containsIgnoreCase(L2_CONF_NAMES, key)) {
				double l2 = TypeUtils.asDouble(c.getValue());
				if (l2 <= 0)
					config.l2(0);
				else 
					config.l2(l2);
			} else if (CollectionUtils.containsIgnoreCase(L1_CONF_NAMES, key)) {
				double l1 = TypeUtils.asDouble(c.getValue());
				if (l1 <= 0)
					config.l1(0);
				else 
					config.l1(l1);
			} else if (CollectionUtils.containsIgnoreCase(LOSS_FUNC_CONF_NAMES, key)) {
				try {
					loss = LossFunction.valueOf(c.getValue().toString());
				} catch (Exception e) {
					LOGGER.debug("Tried to set loss-function using input:" +c.getValue());
					throw new IllegalArgumentException("Invalid LossFunction: "+c.getValue());
				}
			} else if (CollectionUtils.containsIgnoreCase(ACTIVATION_CONF_NAMES, key)) {
				try {
					setActivation(Activation.valueOf(c.getValue().toString()));
				} catch (Exception e) {
					LOGGER.debug("Tried to set activation-function using input:" +c.getValue());
					throw new IllegalArgumentException("Invalid Activation-function: "+c.getValue());
				}
			} 
			
//			else if (CollectionUtils.containsIgnoreCase(N_HIDDEN_CONF_NAMES, key)) {
//				
//			} 
			else {
				LOGGER.debug("Unused Config-argument: " + c);
			}
		}
	}


	
	protected void copyParametersToNew(DL4JMultiLayerBase cpy) {
		cpy.config = config.clone();
		cpy.seed = seed;
		cpy.networkWidth = networkWidth;
		cpy.numHiddenLayers = numHiddenLayers;
		cpy.batchNorm = batchNorm;
		if (explicitLayerWidths != null)
			cpy.explicitLayerWidths = new ArrayList<>(explicitLayerWidths);
		cpy.numEpoch = numEpoch;
		cpy.batchSize = batchSize;
		cpy.saveUpdater = saveUpdater;
		cpy.loss = loss;
		cpy.printInterval = printInterval;
		cpy.dType = dType;
	}
	
	private List<Integer> getWidths(){
		if (explicitLayerWidths!=null && ! explicitLayerWidths.isEmpty()) {
			return explicitLayerWidths;
		} else {
			List<Integer> ws = new ArrayList<>();
			for (int l=0;l<numHiddenLayers; l++) {
				ws.add(networkWidth);
			}
			return ws;
		}
	}
	
	/**
	 * Add all hidden layers
	 * @param bldr The list builder for the network config
	 * @param numIn The input width
	 * @return the width of the last dense layer
	 */
	protected int addHiddenLayers(ListBuilder bldr, int numIn) {
		List<Integer> widths = getWidths();
		
		int previousW = numIn;
		// Hidden layers
		for (int l=0; l<widths.size(); l++) {
			int currW = widths.get(l);
			bldr.layer(new DenseLayer.Builder().nIn(previousW).nOut(currW).build());
			if (batchNorm)
				bldr.layer(new BatchNormalization());
			previousW = currW;
		}
		return previousW;
	}
	

}
