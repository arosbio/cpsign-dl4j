package com.arosbio.ml.dl4j;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.javatuples.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.arosbio.commons.CollectionUtils;
import com.arosbio.commons.TypeUtils;
import com.arosbio.commons.config.BooleanConfigParameter;
import com.arosbio.commons.config.Configurable;
import com.arosbio.commons.config.EnumConfigParameter;
import com.arosbio.commons.config.IntegerConfigParameter;
import com.arosbio.commons.config.NumericConfigParameter;
import com.arosbio.commons.config.StringConfigParameter;
import com.arosbio.commons.config.StringListConfigParameter;
import com.arosbio.ml.nd4j.ND4JUtil.DataConverter;
import com.arosbio.modeling.CPSignSettings;
import com.arosbio.modeling.app.cli.params.converters.IntegerListOrRangeConverter;
import com.arosbio.modeling.app.cli.utils.MultiArgumentSplitter;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.io.PropertyFileSettings;
import com.arosbio.modeling.ml.algorithms.MLAlgorithm;
import com.google.common.collect.Range;
import com.google.common.io.Files;

public abstract class DL4JMultiLayerBase 
	implements MLAlgorithm, Configurable, Closeable {

	private static final Logger LOGGER = LoggerFactory.getLogger(DL4JMultiLayerBase.class);
	private static final String LINE_SEP = System.lineSeparator();

	// Defaults
	public static final int DEFAULT_NUM_HIDDEN_LAYERS = 5;
	public static final int DEFAULT_NETWORK_WIDTH = 10;
	public static final int DEFAULT_N_EPOCH = 10;
	public static final double DEFAULT_TEST_SPLIT_FRAC = 0.1; 
	public static final int DEFAULT_ES_N_EXTRA_EPOCH = 10;
	public static final WeightInit DEFAULT_WEIGHT_INIT = WeightInit.XAVIER;

	//--- Settings - general
	private long seed = CPSignSettings.getInstance().getRNGSeed();
	
	//--- Settings - network structure
	/** The width of the hidden layers in the network - all will have the same width */
	private int networkWidth = DEFAULT_NETWORK_WIDTH;
	private int numHiddenLayers = DEFAULT_NUM_HIDDEN_LAYERS;
	private List<Integer> explicitLayerWidths = null;
	private boolean batchNorm = false;

	//--- Settings - run configs
	private int numEpoch = DEFAULT_N_EPOCH;
	private Integer batchSize;
	private double testSplitFraction = DEFAULT_TEST_SPLIT_FRAC;
	/** Determines for how many extra epochs to run - without improvement in loss score */
	private int earlyStoppingTerminateAfter = 10;
	
	//--- Settings - regularization
	private double inputDropOut = 0;
	private double hiddenLayerDropOut = 0;
	
	
	//--- Settings - only from when instantiated until trained, not loaded again
	protected NeuralNetConfiguration.Builder config;
	protected LossFunctions.LossFunction loss;

	//--- Settings - not required to save
	private transient int evalInterval = 1;
	protected transient DataType dType = DataType.FLOAT;
	private boolean saveUpdater = false;

	//--- Settings only when trained
	/** The input width for the network - should be re-set when model is loaded */
	private transient int inputWidth = -1;
	/** Only != null when model has been trained */
	protected transient MultiLayerNetwork model;


	public DL4JMultiLayerBase(LossFunction lossFunc) {
		this.loss = lossFunc;
		config = new NeuralNetConfiguration.Builder()
				.activation(Activation.RELU)
				.weightInit(DEFAULT_WEIGHT_INIT)
				.updater(new Nesterovs());
	}
	
	public int getInputWidth() {
		return inputWidth;
	}

	public DL4JMultiLayerBase(LossFunction lossFunc, NeuralNetConfiguration.Builder config) {
		this.loss = lossFunc;
		this.config = config;
	}
	
	/*
	 * ****************************************************************************
	 * NETWORK STRUCTURE SETTERS
	 * 
	 * ****************************************************************************
	 */
	
	public DL4JMultiLayerBase networkWidth(int width) {
		this.networkWidth = width;
		return this;
	}
	
	public DL4JMultiLayerBase numHiddenLayers(int nLayers) {
		this.numHiddenLayers = nLayers;
		return this;
	}
	
	/**
	 * Set explicit widths for each hidden layer (i.e. allows different widths for each layer). 
	 * The first index of the array is for the first layer and so on.. If <code>null</code> is sent, the
	 * {@link #networkWidth(int)} and {@link #numHiddenLayers(int)} parameters will be used instead.
	 * @param widths An array of hidden layer widths
	 * @return The calling instance (fluent API)
	 */
	public DL4JMultiLayerBase hiddenLayers(int... widths) {
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
	 * {@link #networkWidth(int)} and {@link #numHiddenLayers(int)} parameters will be used instead.
	 * @param widths A list of hidden layer widths, or <code>null</code> 
	 * @return The calling instance (fluent API)
	 */
	public DL4JMultiLayerBase hiddenLayers(List<Integer> widths) {
		if (widths == null || widths.isEmpty()) {
			explicitLayerWidths = null;
			return this;
		}
		// Use a copy
		this.explicitLayerWidths = new ArrayList<>(widths);
		return this;
	}

	public DL4JMultiLayerBase weightInit(WeightInit init) {
		this.config.weightInit(init);
		return this;
	}
	
	public DL4JMultiLayerBase batchNorm(boolean useBatchNorm) {
		this.batchNorm = useBatchNorm;
		return this;
	}
	
	public DL4JMultiLayerBase lossFunc(LossFunction loss) {
		this.loss = loss;
		return this;
	}

	public DL4JMultiLayerBase activation(IActivation activation) {
		this.config.activation(activation);
		return this;
	}

	public DL4JMultiLayerBase activation(Activation activation) {
		this.config.activation(activation);
		return this;
	}
	
	public DL4JMultiLayerBase dType(DataType type) {
		this.dType = type;
		return this;
	}
	
	/*
	 * ****************************************************************************
	 * RUNNING CONFIG SETTERS
	 * 
	 * ****************************************************************************
	 */

	public DL4JMultiLayerBase numEpoch(int nEpoch) {
		this.numEpoch = nEpoch;
		return this;
	}
	
	public DL4JMultiLayerBase nEpoch(int nEpoch) {
		this.numEpoch = nEpoch;
		return this;
	}
	
	/**
	 * Set a fixed mini-batch size, default is to use 10 mini-batches for each epoch. If setting a negative value 
	 * there will be no mini batches and instead all data will be given for a single batch per epoch.
	 * @param batchSize
	 * @return The instance
	 */
	public DL4JMultiLayerBase batchSize(int batchSize) {
		this.batchSize = batchSize;
		return this;
	}
	
	public DL4JMultiLayerBase testSplitFraction(double testFrac) {
		if (testFrac <0 || testFrac >0.5) {
			LOGGER.debug("Invalid test fraction: " + testFrac);
			throw new IllegalArgumentException("Invalid test split fraction: " + testFrac);
		}
		this.testSplitFraction = testFrac;
		return this;
	}
	
	/**
	 * The updater
	 * @param updater the updater
	 * @return The calling instance (fluent API)
	 */
	public DL4JMultiLayerBase updater(IUpdater updater) {
		this.config.updater(updater);
		return this;
	}
	
	public DL4JMultiLayerBase optimizer(OptimizationAlgorithm alg) {
		this.config.optimizationAlgo(alg);
		return this;
	}
	
	/**
	 * Sets for how many epochs the network should continue training, 
	 * after seeing no improvement in the loss score
	 * @param nEpoch Number of extra epochs, must be &ge; 1
	 * @return The calling instance
	 */
	public DL4JMultiLayerBase earlyStopAfter(int nEpoch) {
		if (nEpoch < 1)
			throw new IllegalArgumentException("Number of epochs to terminate after must be >= 1");
		this.earlyStoppingTerminateAfter = nEpoch;
		return this;
	}
	
	/**
	 * Evaluation interval, perform evaluation of loss score every {@code epoch}
	 * @param epoch how many epochs between evaluation should be done, should be in interval [1..50]
	 * @return The calling instance (fluent API)
	 */
	public DL4JMultiLayerBase evalInterval(int epoch) {
		if (epoch<=0 || epoch>50)
			throw new IllegalArgumentException("Evaluation interval must be within range [1..50]");
		this.evalInterval = epoch;
		return this;
	}
	

	/*
	 * ****************************************************************************
	 * REGULARIZATION SETTERS
	 * 
	 * ****************************************************************************
	 */

	public DL4JMultiLayerBase weightDecay(double decay) {
		config.weightDecay(decay);
		return this;
	}
	
	/**
	 * Set the regularization, either the weightDecay (preferred) or L1/L2 weight penalty.
	 * @param weightDecay a value equal or smaller than zero disable weightDecay regularization
	 * @param l1 a value equal or smaller than zero disable l1 regularization
	 * @param l2 a value equal or smaller than zero disable l2 regularization
	 * @return The calling instance
	 */
	public DL4JMultiLayerBase regularization(double weightDecay, double l1, double l2) {
		if (l1>0)
			config.l1(l1);
		if (l2>0)
			config.l2(l2);
		// Disable (set to 0) if negative weight-decay is given
		config.weightDecay(Math.max(0,weightDecay));
		return this;
	}
	
	/**
	 * Drop out for the hidden layer
	 * @param prob Probprobability of drop-out to occur
	 * @return The calling instance (fluent API)
	 */
	public DL4JMultiLayerBase inputDropOut(double prob) {
		if (prob <0 || prob>=1)
			throw new IllegalArgumentException("Probability of drop-out must be in the range [0..1)");
		this.inputDropOut = prob;
		return this;
	}
	
	/**
	 * Drop out for all hidden layers
	 * @param prob probability of drop-out to occur
	 * @return The calling instance (fluent API)
	 */
	public DL4JMultiLayerBase dropOut(double prob) {
		if (prob <0 || prob>=1)
			throw new IllegalArgumentException("Probability of drop-out must be in the range [0..1)");
		this.hiddenLayerDropOut = prob;
		return this;
	}

	/*
	 * ****************************************************************************
	 * OTHER METHODS
	 * 
	 * ****************************************************************************
	 */

	public Map<String, Object> getProperties() {
		Map<String,Object> p = new HashMap<>();
		p.put(PropertyFileSettings.ML_IMPL_NAME_KEY, getName());
		p.put(PropertyFileSettings.ML_IMPL_KEY, getID());
		p.put("hiddenLayers", getHiddenLayerWidths());
		p.put(PropertyFileSettings.ML_SEED_VALUE_KEY, seed);
		p.put("nEpoch", numEpoch);
		p.put("batchSize", batchSize);
		p.put("usesBatchNorm", batchNorm);
		p.put("internalTestFraction", testSplitFraction);
		p.put("earlyStopTerminateAfter", earlyStoppingTerminateAfter);
		p.put("lossFunc", loss.name());
		return p;
	}

	public void setSeed(long seed) {
		this.seed = seed;
	}

	public long getSeed() {
		return this.seed;
	}

	@Override
	public boolean isFitted() {
		return model != null;
	}

	public abstract MLAlgorithm clone();

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

	@Override
	public void saveToStream(OutputStream ostream) throws IOException, IllegalStateException {
		if (model==null)
			throw new IllegalStateException("Model not trained yet");
		LOGGER.debug("Saving {} model to stream",getName());
		ModelSerializer.writeModel(model, ostream, saveUpdater);
	}

	@Override
	public void loadFromStream(InputStream istream) throws IOException {
		LOGGER.debug("Attempting to load {} model", getName());
		model = ModelSerializer.restoreMultiLayerNetwork(istream);
		inputWidth = model.getLayer(0).getParam("W").rows();
		LOGGER.debug("Finished loading DL4J model with properties: " + model.summary());
	}

	// Structure of the network
	private static List<String> NET_WIDTH_CONF_NAMES = Arrays.asList("width","networkWidth","hiddenLayerWidth");
	private static List<String> N_HIDDEN_CONF_NAMES = Arrays.asList("depth","numHidden","numHiddenLayers");
	private static List<String> HIDDEN_LAYER_WIDTH_NAMES = Arrays.asList("layers","layerWidths");
	private static List<String> WEIGHT_INIT_CONF_NAMES = Arrays.asList("weightInit");
	private static List<String> BATCH_NORM_CONF_NAMES = Arrays.asList("batchNorm");
	private static List<String> LOSS_FUNC_CONF_NAMES = Arrays.asList("loss","lossFunc");
	private static List<String> ACTIVATION_CONF_NAMES = Arrays.asList("activation");
	
	// Running configs
	private static List<String> N_EPOCH_CONF_NAMES = Arrays.asList("nEpoch", "numEpoch");
	private static List<String> BATCH_SIZE_CONF_NAMES = Arrays.asList("batchSize", "miniBatch");
	private static List<String> TEST_FRAC_CONF_NAMES = Arrays.asList("testFrac","internalTestFrac");
	private static List<String> UPDATER_CONF_NAMES = Arrays.asList("updater");
	private static List<String> OPT_CONF_NAMES = Arrays.asList("opt","optimizer");
	private static List<String> EARLY_STOP_AFTER_CONF_NAMES = Arrays.asList("earlyStopAfter"); 
	
	// Regularization
	private static List<String> WEIGHT_DECAY_CONF_NAMES = Arrays.asList("weightDecay");
	private static List<String> L2_CONF_NAMES = Arrays.asList("l2");
	private static List<String> L1_CONF_NAMES = Arrays.asList("l1");
	private static List<String> INPUT_DROP_OUT_CONF_NAMES = Arrays.asList("inputDropOut");
	private static List<String> HIDDEN_DROP_OUT_CONF_NAMES = Arrays.asList("dropOut");
	

	@Override
	public List<ConfigParameter> getConfigParameters() {
		List<ConfigParameter> confs = new ArrayList<>();
		// Network structure
		confs.add(new IntegerConfigParameter(NET_WIDTH_CONF_NAMES, DEFAULT_NETWORK_WIDTH)
				.addDescription("The width of each hidden layer, the input and output-layers are defined by the required input and output"));
		confs.add(new IntegerConfigParameter(N_HIDDEN_CONF_NAMES, DEFAULT_NUM_HIDDEN_LAYERS)
				.addDescription("Number of hidden layers"));
		confs.add(new StringListConfigParameter(HIDDEN_LAYER_WIDTH_NAMES, null)
				.addDescription(String.format("Set custom layer widths for each layer, setting this will ignore the parameters %s and %s",NET_WIDTH_CONF_NAMES.get(0),N_HIDDEN_CONF_NAMES.get(0))));
		confs.add(new EnumConfigParameter<>(WEIGHT_INIT_CONF_NAMES, EnumSet.allOf(WeightInit.class),DEFAULT_WEIGHT_INIT)
				.addDescription("Weight initialization function/distribution. See Deeplearning4j guides for further details."));
		confs.add(new BooleanConfigParameter(BATCH_NORM_CONF_NAMES, false, Arrays.asList(true,false))
				.addDescription("Add batch normalization between all the dense layers"));
		confs.add(new EnumConfigParameter<>(LOSS_FUNC_CONF_NAMES, EnumSet.allOf(LossFunction.class),loss)
				.addDescription("The loss function of the network"));
		confs.add(new EnumConfigParameter<>(ACTIVATION_CONF_NAMES, EnumSet.allOf(Activation.class),Activation.RELU)
				.addDescription("The activation function for the hidden layers"));
		// Running config
		confs.add(new IntegerConfigParameter(N_EPOCH_CONF_NAMES, DEFAULT_N_EPOCH)
				.addDescription("The (maximum) number of epochs to run. Final number could be less depending on if the loss score stops to decrese and the "+EARLY_STOP_AFTER_CONF_NAMES.get(0)+" is not set to a very large number."));
		confs.add(new IntegerConfigParameter(BATCH_SIZE_CONF_NAMES, null)
				.addDescription("The mini-batch size, will control how many records are passed in each iteration. The default is to use 10 mini-batches for each epoch "
						+ "- thus calculated at training-time. Setting a value smaller than 0 disables mini-batches and passes _all data_ through at the same time (i.e., one batch per epoch)."));
		confs.add(new NumericConfigParameter(TEST_FRAC_CONF_NAMES, DEFAULT_TEST_SPLIT_FRAC, Range.closed(0., .5))
				.addDescription("Fraction of training examples that should be used for monitoring improvements of the network during training, these will not be used in the training of the network. If there are examples in this internal test-set, these will be used for determining early stopping - otherwise all training examples loss scores are used instead."));
		confs.add(new StringConfigParameter(UPDATER_CONF_NAMES, "Nesterovs,"+Nesterovs.DEFAULT_NESTEROV_LEARNING_RATE+","+Nesterovs.DEFAULT_NESTEROV_MOMENTUM)
				.addDescription("Set the updater (https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/troubleshooting-training#updater-and-optimization-algorithm). "
						+ "Using the Java API this can be concrete instances of the IUpdater interface, including scheduling for e.g. learning rate. For CLI users no scheduling is possible and the "
						+ "following syntaxes are possible to use (replace the <param> with concrete floating point values);"+LINE_SEP+DL4JUtils.supportedUpdaterInput()));
		confs.add(new EnumConfigParameter<>(OPT_CONF_NAMES, EnumSet.allOf(OptimizationAlgorithm.class), OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.addDescription("The optmization algorithm, for furhter info refer to: https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/troubleshooting-training#updater-and-optimization-algorithm"));
		confs.add(new IntegerConfigParameter(EARLY_STOP_AFTER_CONF_NAMES, DEFAULT_ES_N_EXTRA_EPOCH, Range.atLeast(1))
				.addDescription("Determines how many epochs to continue to run without having an improvement in the loss function. If there should be no early stopping (always run all specified epochs) specify the same number as that of parameter "+N_EPOCH_CONF_NAMES.get(0)));
		// Regularization
		confs.add(new NumericConfigParameter(WEIGHT_DECAY_CONF_NAMES, 0)
				.addDescription("The weight-decay regularization term, put to <=0 if not to use weight-decay regularization"));
		confs.add(new NumericConfigParameter(L2_CONF_NAMES, 0)
				.addDescription("L2 regularization term. Disabled by default."));
		confs.add(new NumericConfigParameter(L1_CONF_NAMES, 0)
				.addDescription("L1 regularization term. Disabled by default."));
		confs.add(new NumericConfigParameter(INPUT_DROP_OUT_CONF_NAMES, 0,Range.closedOpen(0., 1.))
				.addDescription("Drop-out rate for the input layer. 0 means no drop-out, 0.5 is 50%% chance of drop-out etc."));
		confs.add(new NumericConfigParameter(HIDDEN_DROP_OUT_CONF_NAMES, 0,Range.closedOpen(0., 1.))
				.addDescription("Drop-out rate for the hidden layers. 0 means no drop-out, 0.5 is 50%% chance of drop-out etc."));
		
		return confs;
	}

	@Override
	public void setConfigParameters(Map<String, Object> params) 
			throws IllegalStateException, IllegalArgumentException {

		for (Map.Entry<String, Object> c : params.entrySet()) {
			String key = c.getKey();
			// Network structure 
			if (CollectionUtils.containsIgnoreCase(NET_WIDTH_CONF_NAMES, key)) {
				networkWidth = TypeUtils.asInt(c.getValue());
			} else if (CollectionUtils.containsIgnoreCase(N_HIDDEN_CONF_NAMES, key)) {
				numHiddenLayers = TypeUtils.asInt(c.getValue());
			} else if (CollectionUtils.containsIgnoreCase(HIDDEN_LAYER_WIDTH_NAMES, key)) {
				try {
					List<Integer> tmp = new ArrayList<>();
					if (c.getValue() == null || c.getValue().toString().equalsIgnoreCase("null")) {
						// Reset widths 
						explicitLayerWidths = null;
						LOGGER.debug("Configured to removed explicit layers widths");
						continue;
					} else if (c.getValue() instanceof List) {
						LOGGER.debug("Configuring explicit layers using a List input");

						for (Object o : (List<?>)c.getValue()) {
							tmp.add(TypeUtils.asInt(o));
						}

					} else if (c.getValue() instanceof Object[]) {
						for (Object o : (Object[])c.getValue()) {
							tmp.add(TypeUtils.asInt(o));
						}
					} else if (c.getValue() instanceof int[]){
						for (int w : (int[])c.getValue()) {
							tmp.add(w);
						}
					} else if (c.getValue() instanceof String) {
						List<String> splits = MultiArgumentSplitter.split(c.getValue().toString());
						for (String s : splits) {
							tmp.addAll(new IntegerListOrRangeConverter().convert(s));
						}
					} else {
						LOGGER.debug("Invalid argument type for parameter {}: {}",HIDDEN_LAYER_WIDTH_NAMES,c.getValue());
						throw new IllegalArgumentException("Invalid argument for config parameter " + HIDDEN_LAYER_WIDTH_NAMES.get(0));
					}

					LOGGER.debug("Collected widths: " + tmp);
					// Verify all withs are >0
					for (int w : tmp) {
						if (w <= 0)
							throw new IllegalArgumentException("Invalid argument for config "+HIDDEN_LAYER_WIDTH_NAMES.get(0) + ": widths cannot be <=0, got: " + w);
					}

					// Here configuration was successful, set the updated list
					explicitLayerWidths = tmp;
					LOGGER.debug("Configuring explicit list of hidden layer widths, using final list: {}",explicitLayerWidths);
				} catch (IllegalArgumentException e) {
					// This should probably be formatted correctly - pass it along
					throw e;
				} catch (Exception e) {
					LOGGER.debug("Invalid argument for {}: {}",HIDDEN_LAYER_WIDTH_NAMES, c.getValue());
					throw new IllegalArgumentException("Invalid input for config " + HIDDEN_LAYER_WIDTH_NAMES.get(0) + ": " + c.getValue());
				}
			} else if (CollectionUtils.containsIgnoreCase(WEIGHT_INIT_CONF_NAMES, key)){
				try {
					weightInit(WeightInit.valueOf(c.getValue().toString()));
				} catch (Exception e) {
					LOGGER.debug("Tried to set weightInit using input: {}",c.getValue());
					throw new IllegalArgumentException("Invalid weightInit value: "+c.getValue());
				}
			} else if (CollectionUtils.containsIgnoreCase(BATCH_NORM_CONF_NAMES, key)) {
				batchNorm = TypeUtils.asBoolean(c.getValue());
			} else if (CollectionUtils.containsIgnoreCase(LOSS_FUNC_CONF_NAMES, key)) {
				try {
					loss = LossFunction.valueOf(c.getValue().toString());
				} catch (Exception e) {
					LOGGER.debug("Tried to set loss-function using input:{}", c.getValue());
					throw new IllegalArgumentException("Invalid LossFunction: "+c.getValue());
				}
			} else if (CollectionUtils.containsIgnoreCase(ACTIVATION_CONF_NAMES, key)) {
				try {
					activation(Activation.valueOf(c.getValue().toString()));
				} catch (Exception e) {
					LOGGER.debug("Tried to set activation-function using input: {}",c.getValue());
					throw new IllegalArgumentException("Invalid Activation-function: "+c.getValue());
				}
			}
			// Run configs
			else if (CollectionUtils.containsIgnoreCase(N_EPOCH_CONF_NAMES, key)) {
				numEpoch = TypeUtils.asInt(c.getValue());
			} else if (CollectionUtils.containsIgnoreCase(BATCH_SIZE_CONF_NAMES, key)) {
				if (c.getValue() == null)
					batchSize = null;
				else
					batchSize = TypeUtils.asInt(c.getValue());
			} else if (CollectionUtils.containsIgnoreCase(TEST_FRAC_CONF_NAMES, key)) {
				testSplitFraction(TypeUtils.asDouble(c.getValue()));
			} else if (CollectionUtils.containsIgnoreCase(UPDATER_CONF_NAMES, key)) {
				Object val = c.getValue();
				if (val instanceof IUpdater) {
					updater((IUpdater) val);
				} else if (val instanceof String) {
					updater(DL4JUtils.toUpdater((String)val));
				} else {
					LOGGER.debug("Invalid input for updater, class={}, str={}",val.getClass(),val.toString());
					throw new IllegalArgumentException("Invalid updater: " + val.toString());
				}
			} else if (CollectionUtils.containsIgnoreCase(OPT_CONF_NAMES, key)) {
				try {
					optimizer(OptimizationAlgorithm.valueOf(c.getValue().toString()));
				} catch (Exception e) {
					LOGGER.debug("Tried to set optimization algorithm using input:" +c.getValue());
					throw new IllegalArgumentException("Invalid Optimization algorithm: "+c.getValue());
				}
			} else if (CollectionUtils.containsIgnoreCase(EARLY_STOP_AFTER_CONF_NAMES, key)) {
				earlyStopAfter(TypeUtils.asInt(c.getValue()));
			} 
			// Regularization
			else if (CollectionUtils.containsIgnoreCase(WEIGHT_DECAY_CONF_NAMES, key)) {
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
			} else if (CollectionUtils.containsIgnoreCase(INPUT_DROP_OUT_CONF_NAMES, key)) { 
				inputDropOut(TypeUtils.asDouble(c.getValue()));
			} else if (CollectionUtils.containsIgnoreCase(HIDDEN_DROP_OUT_CONF_NAMES, key)) {
				dropOut(TypeUtils.asDouble(c.getValue()));
			}  else {
				LOGGER.debug("Unused Config-argument: " + c);
			}
		}
	}




	protected void copyParametersToNew(DL4JMultiLayerBase cpy) {
		cpy.config = config.clone();
		cpy.seed = seed;
		cpy.networkWidth = networkWidth;
		cpy.numHiddenLayers = numHiddenLayers;
		if (explicitLayerWidths != null)
			cpy.explicitLayerWidths = new ArrayList<>(explicitLayerWidths);
		cpy.numEpoch = numEpoch;
		cpy.batchSize = batchSize;
		cpy.batchNorm = batchNorm;
		cpy.testSplitFraction = testSplitFraction;
		cpy.saveUpdater = saveUpdater;
		cpy.loss = loss;
		cpy.evalInterval = evalInterval;
		cpy.dType = dType;
	}

	protected Pair<List<DataRecord>,List<DataRecord>> getInternalTrainTestSplits(List<DataRecord> allRecs){
		if (testSplitFraction <= 0)
			return Pair.with(allRecs, new ArrayList<>());

		List<DataRecord> cpy = new ArrayList<>(allRecs);

		int testSplitIndex = (int) (allRecs.size()*testSplitFraction);
		List<DataRecord> test = cpy.subList(0, testSplitIndex);
		List<DataRecord> train = cpy.subList(testSplitIndex, allRecs.size());

		return Pair.with(train, test);
	}

	protected void trainNetwork(MultiLayerConfiguration config, 
			List<DataRecord> trainingData,
			boolean isClassification) {
		// Create a tmp dir to save intermediate models
		File tmpDir = Files.createTempDir();

		Pair<List<DataRecord>,List<DataRecord>> trainTest = getInternalTrainTestSplits(trainingData);
		DataConverter trainConv = isClassification ? DataConverter.classification(trainTest.getValue0()) : DataConverter.regression(trainTest.getValue0());
		// calculate batch size and 
		int batch = calcBatchSize(trainTest.getValue0().size());
		DataSetIterator trainIter = new INDArrayDataSetIterator(trainConv, batch);

		// Generate Test set
		DataSetIterator testIter = null;
		if (testSplitFraction >0) {
			DataConverter testConv = isClassification ? DataConverter.classification(trainTest.getValue1(), trainConv.getNumAttributes(), trainConv.getOneHotMapping()) : DataConverter.regression(trainTest.getValue1(),trainConv.getNumAttributes());
			testIter = new INDArrayDataSetIterator(testConv, batch);
		} else {
			testIter = new INDArrayDataSetIterator(trainConv, batch);
		}
		
		EarlyStoppingModelSaver<MultiLayerNetwork> saver = new LocalFileModelSaver(tmpDir);
		EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
				.epochTerminationConditions(new MaxEpochsTerminationCondition(numEpoch), // Max num epochs
						new ScoreImprovementEpochTerminationCondition(earlyStoppingTerminateAfter,1E-5)) // Check for improvement
				.evaluateEveryNEpochs(evalInterval) // Evaluate every 1-50 epoch
				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) // Max of 20 minutes
				.scoreCalculator(new DataSetLossCalculator(testIter, true)) // Calculate test set score
				.modelSaver(saver)
				.build();
		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,config,trainIter);

		// Conduct early stopping training
		EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

		// Debug some metrics
		LOGGER.debug("Finished training using early stopping, metrics{}  termination reason: {}{}  termination details: {}{}  total epochs: {}/{}{}  best epoch: {}",
				LINE_SEP, result.getTerminationReason(),LINE_SEP, result.getTerminationDetails(),LINE_SEP, result.getTotalEpochs(),numEpoch,LINE_SEP, result.getBestModelEpoch());

		// Log scores 
		StringBuilder scoreOutput = new StringBuilder();
		if (testSplitFraction>0)
			writeLossScores(null, result.getScoreVsEpoch(), scoreOutput);
		else
			writeLossScores(result.getScoreVsEpoch(), null, scoreOutput);
		LOGGER.debug("Scores produced during training of the network:{}{}",LINE_SEP, scoreOutput.toString());
		
		if (Double.isNaN(result.getBestModelScore())) {
			LOGGER.debug("The best score was NaN - so the model should be bad - throwing an exception");
			throw new IllegalArgumentException("The "+getName() + " model could not be fitted using the current parameters - please revise them");
		}
		
		model = result.getBestModel();
		// Set the labels
		if (isClassification)
			model.setLabels(trainConv.getOneHotMapping().getLabelsND());
	}

	/**
	 * Write scores to an {@link Appendable} object
	 * @param trainScores Scores for train instances
	 * @param testScores Scores for test instances
	 * @param a Appendable to write to
	 * @return {@code true} if anything was written, {@code false} otherwise
	 */
	public static boolean writeLossScores(Map<Integer,Double> trainScores, Map<Integer,Double> testScores, Appendable a){
		boolean hasTrainS = trainScores != null && !trainScores.isEmpty();
		boolean hasTestS = testScores != null && !testScores.isEmpty();
		if (!hasTestS && !hasTrainS) {
			return false;
		}
		try (
				CSVPrinter printer = CSVFormat.DEFAULT.print(a);){
			// Print header
			printer.print("Epoch");
			if (hasTrainS)
				printer.print("Train Score");
			if (hasTestS)
				printer.print("Test Score");
			printer.println();

			// Print score(s) vs. epoch
			List<Integer> list = new ArrayList<>();
			if (hasTestS) list.addAll(testScores.keySet());
			if (hasTrainS) list.addAll(trainScores.keySet());
			Collections.sort(list);
			for (int e : list) {
				printer.print(e);
				if (hasTrainS) printer.print(getValue(trainScores, e));
				if (hasTestS) printer.print(getValue(testScores, e));
				printer.println();
			}
		} catch (IOException e) {
			LOGGER.error("Failed generating CSV with scores from training, reason: {}",e.getMessage());
			LOGGER.debug("exception",e);
			return false;
		}
		return true;
	}

	private static String getValue(Map<Integer,Double> scores,int epoch) {
		Double s = scores.getOrDefault(epoch, null);
		return s != null? s.toString() : "-";
	}

	protected List<Integer> getHiddenLayerWidths(){
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
		List<Integer> widths = getHiddenLayerWidths();
		if (widths == null || widths.isEmpty()) {
			throw new IllegalStateException("Need to have at least one hidden layer: please revise network parameters");
		}
		
		inputWidth = numIn;
		
		// If input drop out should be applied
		if (inputDropOut>0)
			bldr.layer(new DropoutLayer(1-inputDropOut));

		int previousW = numIn;
		// Hidden layers
		for (int l=0; l<widths.size(); l++) {
			int currW = widths.get(l);
			bldr.layer(new DenseLayer.Builder().nIn(previousW).nOut(currW).build());
			if (hiddenLayerDropOut>0)
				bldr.layer(new DropoutLayer(1-hiddenLayerDropOut));
			if (batchNorm)
				bldr.layer(new BatchNormalization());
			previousW = currW;
		}
		return previousW;
	}

	/**
	 * Closes the underlying network and frees all resources
	 */
	public void close() {
		if (model != null)
			model.close();
	}

}
