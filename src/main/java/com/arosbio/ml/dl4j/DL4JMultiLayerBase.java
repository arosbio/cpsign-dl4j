package com.arosbio.ml.dl4j;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.BaseEarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
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
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
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
import com.arosbio.ml.dl4j.eval.EarlyStopScoreListener;
import com.arosbio.ml.dl4j.eval.EarlyStopScoreListenerFileWrite;
import com.arosbio.ml.nd4j.ND4JUtil.DataConverter;
import com.arosbio.modeling.CPSignSettings;
import com.arosbio.modeling.app.cli.params.converters.IntegerListOrRangeConverter;
import com.arosbio.modeling.app.cli.utils.MultiArgumentSplitter;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.io.PropertyFileSettings;
import com.arosbio.modeling.ml.algorithms.MLAlgorithm;
import com.google.common.collect.Range;
import com.google.common.io.Files;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public abstract class DL4JMultiLayerBase 
implements MLAlgorithm, Configurable, Closeable {

	private static final Logger LOGGER = (Logger) LoggerFactory.getLogger(DL4JMultiLayerBase.class);
	private static final String LINE_SEP = System.lineSeparator();

	static {
		Logger earlyStopLogger = (Logger) LoggerFactory.getLogger(BaseEarlyStoppingTrainer.class);
		Logger dataIterLogger = (Logger) LoggerFactory.getLogger(AsyncDataSetIterator.class);
		Logger scoreImproveCondLogger = (Logger) LoggerFactory.getLogger(ScoreImprovementEpochTerminationCondition.class);
		earlyStopLogger.setLevel(Level.ERROR);
		dataIterLogger.setLevel(Level.INFO);
		scoreImproveCondLogger.setLevel(Level.WARN);
	}

	// Defaults
	public static final int DEFAULT_NUM_HIDDEN_LAYERS = 5;
	public static final int DEFAULT_NETWORK_WIDTH = 10;
	public static final int DEFAULT_N_EPOCH = 10;
	public static final double DEFAULT_TEST_SPLIT_FRAC = 0.1; 
	public static final int DEFAULT_ES_N_EXTRA_EPOCH = 10;
	public static final WeightInit DEFAULT_WEIGHT_INIT = WeightInit.XAVIER;
	public static final int DEFAULT_ITER_TIMEOUT_MINS = 20;

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
	private int iterationTimeoutMins = DEFAULT_ITER_TIMEOUT_MINS;
	/** A path that can be resolved as absolute, user-relative or relative */
	private String scoresOutputFile;

	//--- Settings - regularization
	private transient double l1=0, l2=0, weightDecay=0;
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

	/*
	 * ****************************************************************************
	 * CONSTRUCTORS
	 * 
	 * ****************************************************************************
	 */
	public DL4JMultiLayerBase(LossFunction lossFunc) {
		this.loss = lossFunc;
		config = new NeuralNetConfiguration.Builder()
				.activation(Activation.RELU)
				.weightInit(DEFAULT_WEIGHT_INIT)
				.updater(new Nesterovs());
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

	public DL4JMultiLayerBase gradientNorm(GradientNormalization norm) {
		if (norm == null)
			config.gradientNormalization(GradientNormalization.None);
		else
			config.gradientNormalization(norm);
		return this;
	}

	public DL4JMultiLayerBase gradientNormalization(GradientNormalization norm) {
		return gradientNorm(norm);
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
	
	public DL4JMultiLayerBase iterationTerminationTimeout(int minutes) {
		if (minutes < 1)
			throw new IllegalArgumentException("Iteration timeout must be at least 1 minute");
		this.iterationTimeoutMins = minutes;
		return this;
	}

	public DL4JMultiLayerBase lossOutput(String path) {
		this.scoresOutputFile = path;
		return this;
	}

	public DL4JMultiLayerBase lossScores(String path) {
		return lossOutput(path);
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
		// If no previous weightDecay was set and no new weightDecay - make no changes
		if (decay <=0 && this.weightDecay <= 0)
			return this;
		this.weightDecay = Math.max(0, decay);
		config.weightDecay(this.weightDecay);
		return this;
	}

	public DL4JMultiLayerBase l1(double l1) {
		// If no previous l1 was set and no new l1 - make no changes
		if (l1 <=0 && this.l1 <= 0)
			return this;
		this.l1 = Math.max(0,l1); 
		config.l1(this.l1);
		return this;
	}

	public DL4JMultiLayerBase l2(double l2) {
		// If no previous l2 was set and no new l2 - make no changes
		if (l2 <=0 && this.l2 <= 0)
			return this;
		this.l2 = Math.max(0,l2);
		config.l2(this.l2);
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
		l1(l1);
		l2(l2);
		weightDecay(weightDecay);
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

	public int getInputWidth() {
		return inputWidth;
	}

	public Map<String, Object> getProperties() {
		Map<String,Object> p = new HashMap<>();
		// General CPSign stuff
		p.put(PropertyFileSettings.ML_IMPL_NAME_KEY, getName());
		p.put(PropertyFileSettings.ML_IMPL_KEY, getID());
		p.put(PropertyFileSettings.ML_SEED_VALUE_KEY, seed);

		// Network structure
		p.put(HIDDEN_LAYER_WIDTH_CONF_NAMES.get(0), toCPSignConfigList(getHiddenLayerWidths()));
		try {
			p.put(WEIGHT_INIT_CONF_NAMES.get(0), DL4JUtils.reverseLookup(config.getWeightInitFn()).toString()); 
		} catch (Exception e) {
			LOGGER.warn("Could not set the correct {} parameter in the properties",WEIGHT_INIT_CONF_NAMES.get(0));
		}
		p.put(BATCH_NORM_CONF_NAMES.get(0), batchNorm);
		p.put(GRAD_NORM_CONF_NAMES.get(0), config.getGradientNormalization().name());
		p.put(LOSS_FUNC_CONF_NAMES.get(0), loss.name());
		p.put(ACTIVATION_CONF_NAMES.get(0), DL4JUtils.reverseLookup(config.getActivationFn()).toString()); 

		// Run configs
		p.put(N_EPOCH_CONF_NAMES.get(0), numEpoch);
		// only add if != null
		if (batchSize != null)
			p.put(BATCH_SIZE_CONF_NAMES.get(0), batchSize);
		p.put(TEST_FRAC_CONF_NAMES.get(0), testSplitFraction);
		p.put(UPDATER_CONF_NAMES.get(0), DL4JUtils.toString(config.getIUpdater()));
		p.put(OPT_CONF_NAMES.get(0), config.getOptimizationAlgo().toString());
		p.put(EARLY_STOP_AFTER_CONF_NAMES.get(0), earlyStoppingTerminateAfter);
		p.put(ITERATION_TIMEOUT_CONF_NAMES.get(0), iterationTimeoutMins);
		if (scoresOutputFile != null)
			p.put(TRAIN_LOSS_FILE_PATH_CONF_NAMES.get(0), scoresOutputFile);

		// Regularization
		p.put(WEIGHT_DECAY_CONF_NAMES.get(0), weightDecay);
		p.put(L2_CONF_NAMES.get(0), l2);
		p.put(L1_CONF_NAMES.get(0), l1);
		p.put(INPUT_DROP_OUT_CONF_NAMES.get(0), inputDropOut);
		p.put(HIDDEN_DROP_OUT_CONF_NAMES.get(0), hiddenLayerDropOut);

		return p;
	}

	private static String toCPSignConfigList(List<?> list) {
		if (list == null || list.isEmpty())
			return "";
		StringBuilder sb = new StringBuilder();
		for (int i=0; i<list.size()-1; i++) {
			sb.append(list.get(i)).append(',');
		}
		// Add last index
		sb.append(list.get(list.size()-1));
		return sb.toString();
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
	private static List<String> HIDDEN_LAYER_WIDTH_CONF_NAMES = Arrays.asList("layers","layerWidths");
	private static List<String> WEIGHT_INIT_CONF_NAMES = Arrays.asList("weightInit");
	private static List<String> BATCH_NORM_CONF_NAMES = Arrays.asList("batchNorm");
	private static List<String> GRAD_NORM_CONF_NAMES = Arrays.asList("gradNorm","gradientNorm");
	private static List<String> LOSS_FUNC_CONF_NAMES = Arrays.asList("loss","lossFunc");
	private static List<String> ACTIVATION_CONF_NAMES = Arrays.asList("activation");

	// Running configs
	private static List<String> N_EPOCH_CONF_NAMES = Arrays.asList("nEpoch", "numEpoch");
	private static List<String> BATCH_SIZE_CONF_NAMES = Arrays.asList("batchSize", "miniBatch");
	private static List<String> TEST_FRAC_CONF_NAMES = Arrays.asList("testFrac","internalTestFrac");
	private static List<String> UPDATER_CONF_NAMES = Arrays.asList("updater");
	private static List<String> OPT_CONF_NAMES = Arrays.asList("opt","optimizer");
	private static List<String> EARLY_STOP_AFTER_CONF_NAMES = Arrays.asList("earlyStopAfter");
	private static List<String> ITERATION_TIMEOUT_CONF_NAMES = Arrays.asList("iterTimeout","iterationTimeout");
	private static List<String> TRAIN_LOSS_FILE_PATH_CONF_NAMES = Arrays.asList("lossScoresOutput","trainOutput");

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
		confs.add(new StringListConfigParameter(HIDDEN_LAYER_WIDTH_CONF_NAMES, null)
				.addDescription(String.format("Set custom layer widths for each layer, setting this will ignore the parameters %s and %s",NET_WIDTH_CONF_NAMES.get(0),N_HIDDEN_CONF_NAMES.get(0))));
		confs.add(new EnumConfigParameter<>(WEIGHT_INIT_CONF_NAMES, EnumSet.allOf(WeightInit.class),DEFAULT_WEIGHT_INIT)
				.addDescription("Weight initialization function/distribution. See Deeplearning4j guides for further details."));
		confs.add(new BooleanConfigParameter(BATCH_NORM_CONF_NAMES, false, Arrays.asList(true,false))
				.addDescription("Add batch normalization between all the dense layers"));
		confs.add(new EnumConfigParameter<>(GRAD_NORM_CONF_NAMES, EnumSet.allOf(GradientNormalization.class), GradientNormalization.None)
				.addDescription("Add gradient normalization, applied to all layers"));
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
		confs.add(new StringConfigParameter(UPDATER_CONF_NAMES, "Nesterovs;"+Nesterovs.DEFAULT_NESTEROV_LEARNING_RATE+";"+Nesterovs.DEFAULT_NESTEROV_MOMENTUM)
				.addDescription("Set the updater (https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/troubleshooting-training#updater-and-optimization-algorithm). "
						+ "Using the Java API this can be concrete instances of the IUpdater interface, including scheduling for e.g. learning rate. For CLI users no scheduling is possible and the "
						+ "following syntaxes are possible to use (replace the <param> with concrete floating point values);"+LINE_SEP+DL4JUtils.supportedUpdaterInput()));
		confs.add(new EnumConfigParameter<>(OPT_CONF_NAMES, EnumSet.allOf(OptimizationAlgorithm.class), OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.addDescription("The optmization algorithm, for furhter info refer to: https://deeplearning4j.konduit.ai/deeplearning4j/how-to-guides/tuning-and-training/troubleshooting-training#updater-and-optimization-algorithm"));
		confs.add(new IntegerConfigParameter(EARLY_STOP_AFTER_CONF_NAMES, DEFAULT_ES_N_EXTRA_EPOCH, Range.atLeast(1))
				.addDescription("Determines how many epochs to continue to run without having an improvement in the loss function. If there should be no early stopping (always run all specified epochs) specify the same number as that of parameter "+N_EPOCH_CONF_NAMES.get(0)));
		confs.add(new IntegerConfigParameter(ITERATION_TIMEOUT_CONF_NAMES, DEFAULT_ITER_TIMEOUT_MINS)
				.addDescription("Specify a termination criterion for how long a single iteration (i.e. one batch passed through+backprop). Specified as the maximum number of minutes for a single interation."));
		confs.add(new StringConfigParameter(TRAIN_LOSS_FILE_PATH_CONF_NAMES, null)
				.addDescription("Specify a file or directory to print loss-scores from the training epochs to (in csv format), default is otherwise to print them in the logfile"));
		// Regularization
		confs.add(new NumericConfigParameter(WEIGHT_DECAY_CONF_NAMES, 0)
				.addDescription("The weight-decay regularization term, put to <=0 if not to use weight-decay regularization. Note that L2 and weightDecay cannot be used together."));
		confs.add(new NumericConfigParameter(L2_CONF_NAMES, 0)
				.addDescription("L2 regularization term. Disabled by default. Note that L2 and weightDecay cannot be used together."));
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
				networkWidth(TypeUtils.asInt(c.getValue()));
			} else if (CollectionUtils.containsIgnoreCase(N_HIDDEN_CONF_NAMES, key)) {
				numHiddenLayers(TypeUtils.asInt(c.getValue()));
			} else if (CollectionUtils.containsIgnoreCase(HIDDEN_LAYER_WIDTH_CONF_NAMES, key)) {
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
						LOGGER.debug("Invalid argument type for parameter {}: {}",HIDDEN_LAYER_WIDTH_CONF_NAMES,c.getValue());
						throw new IllegalArgumentException("Invalid argument for config parameter " + HIDDEN_LAYER_WIDTH_CONF_NAMES.get(0));
					}

					LOGGER.debug("Collected widths: " + tmp);
					// Verify all withs are >0
					for (int w : tmp) {
						if (w <= 0)
							throw new IllegalArgumentException("Invalid argument for config "+HIDDEN_LAYER_WIDTH_CONF_NAMES.get(0) + ": widths cannot be <=0, got: " + w);
					}

					// Here configuration was successful, set the updated list
					explicitLayerWidths = tmp;
					LOGGER.debug("Configuring explicit list of hidden layer widths, using final list: {}",explicitLayerWidths);
				} catch (IllegalArgumentException e) {
					// This should probably be formatted correctly - pass it along
					throw e;
				} catch (Exception e) {
					LOGGER.debug("Invalid argument for {}: {}",HIDDEN_LAYER_WIDTH_CONF_NAMES, c.getValue());
					throw new IllegalArgumentException("Invalid input for config " + HIDDEN_LAYER_WIDTH_CONF_NAMES.get(0) + ": " + c.getValue());
				}
			} else if (CollectionUtils.containsIgnoreCase(WEIGHT_INIT_CONF_NAMES, key)){
				try {
					weightInit(WeightInit.valueOf(c.getValue().toString()));
				} catch (Exception e) {
					LOGGER.debug("Tried to set weightInit using input: {}",c.getValue());
					throw new IllegalArgumentException("Invalid weightInit value: "+c.getValue());
				}
			} else if (CollectionUtils.containsIgnoreCase(BATCH_NORM_CONF_NAMES, key)) {
				batchNorm(TypeUtils.asBoolean(c.getValue()));
			} else if (CollectionUtils.containsIgnoreCase(GRAD_NORM_CONF_NAMES, key)) {
				if (c.getValue()==null) {
					gradientNorm(null);
				} else if (c.getValue() instanceof GradientNormalization) {
					gradientNorm((GradientNormalization)c.getValue());
				} else if (c.getValue() instanceof String){
					gradientNorm(GradientNormalization.valueOf(c.getValue().toString()));
				} else {
					throw new IllegalArgumentException("Invalid gradientNorm value: " + c.getValue());
				}
			} else if (CollectionUtils.containsIgnoreCase(LOSS_FUNC_CONF_NAMES, key)) {
				try {
					lossFunc(LossFunction.valueOf(c.getValue().toString()));
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
				numEpoch(TypeUtils.asInt(c.getValue()));
			} else if (CollectionUtils.containsIgnoreCase(BATCH_SIZE_CONF_NAMES, key)) {
				if (c.getValue() == null || "null".equals(c.getValue().toString()))
					batchSize = null;
				else
					batchSize(TypeUtils.asInt(c.getValue()));
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
			} else if (CollectionUtils.containsIgnoreCase(ITERATION_TIMEOUT_CONF_NAMES, key)) {
				iterationTerminationTimeout(TypeUtils.asInt(c.getValue()));
			} else if (CollectionUtils.containsIgnoreCase(TRAIN_LOSS_FILE_PATH_CONF_NAMES, key)) {
				if (c.getValue() == null || c.getValue() instanceof String)
					lossOutput((String)c.getValue());
				else 
					throw new IllegalArgumentException("Invalid input type for parameter "+TRAIN_LOSS_FILE_PATH_CONF_NAMES.get(0));
			}


			// Regularization
			else if (CollectionUtils.containsIgnoreCase(WEIGHT_DECAY_CONF_NAMES, key)) {
				double decay = TypeUtils.asDouble(c.getValue());
				weightDecay(decay);
			} else if (CollectionUtils.containsIgnoreCase(L2_CONF_NAMES, key)) {
				double l2 = TypeUtils.asDouble(c.getValue());
				l2(l2);
			} else if (CollectionUtils.containsIgnoreCase(L1_CONF_NAMES, key)) {
				double l1 = TypeUtils.asDouble(c.getValue());
				l1(l1);
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
		cpy.batchNorm = batchNorm;

		cpy.numEpoch = numEpoch;
		cpy.batchSize = batchSize;
		cpy.testSplitFraction = testSplitFraction;
		cpy.earlyStoppingTerminateAfter = earlyStoppingTerminateAfter;
		cpy.iterationTimeoutMins = iterationTimeoutMins;
		cpy.scoresOutputFile = scoresOutputFile;

		cpy.inputDropOut = inputDropOut;
		cpy.hiddenLayerDropOut = hiddenLayerDropOut;

		cpy.loss = loss;
		cpy.evalInterval = evalInterval;

		cpy.dType = dType;
		cpy.saveUpdater = saveUpdater;
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

	protected void trainNetwork(MultiLayerConfiguration nnConfig, 
			List<DataRecord> trainingData,
			boolean isClassification) throws IllegalArgumentException{
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
			// Check if batch size is OK for the test-set (that should be much smaller), otherwise pass all records at the same time
			int nTestEx = trainTest.getValue1().size();
			int testBatchSize = (nTestEx / batch) >= 2 ? batch : nTestEx;
			testIter = new INDArrayDataSetIterator(testConv, testBatchSize);
		} else {
			testIter = new INDArrayDataSetIterator(trainConv, batch);
		}

		EarlyStoppingModelSaver<MultiLayerNetwork> saver = new LocalFileModelSaver(tmpDir);
		EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
				.epochTerminationConditions(new MaxEpochsTerminationCondition(numEpoch), // Max num epochs
						new ScoreImprovementEpochTerminationCondition(earlyStoppingTerminateAfter,1E-5)) // Check for improvement
				.evaluateEveryNEpochs(evalInterval) // Evaluate every epochInterval 
				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(iterationTimeoutMins, TimeUnit.MINUTES)) // Max time per iteration
				.scoreCalculator(new DataSetLossCalculator(testIter, true)) // Calculate test/train set score
				.modelSaver(saver)
				.build();

		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,nnConfig,trainIter);

		// Set up the score listener
		StringBuilder scoreOutput = new StringBuilder();
		try {
			EarlyStopScoreListener listener = scoresOutputFile != null ? new EarlyStopScoreListenerFileWrite(scoresOutputFile, testSplitFraction>0,getProperties()) : new EarlyStopScoreListener(scoreOutput, testSplitFraction>0,getProperties());
			if (testSplitFraction>0) {
				// If using test-examples for scoring - also add train scores 
				listener.trainScorer(new DataSetLossCalculator(trainIter, true));
			}
			trainer.setListener(listener);
		} catch (IOException e) {
			LOGGER.debug("Failed setting up score listener",e);
			throw new IllegalArgumentException(e);
		}


		// Conduct early stopping training
		EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

		// Debug some metrics
		LOGGER.debug("Finished training using early stopping, metrics{}  termination reason: {}{}  termination details: {}{}  total epochs: {}/{}{}  best epoch: {}",
				LINE_SEP, result.getTerminationReason(),LINE_SEP, result.getTerminationDetails(),LINE_SEP, result.getTotalEpochs(),numEpoch,LINE_SEP, result.getBestModelEpoch());

		// Log scores in case not written to file
		if (scoreOutput.length()>0) {
			LOGGER.debug("Scores produced during training of the network:{}{}",LINE_SEP, scoreOutput.toString());
		}

		if (Double.isNaN(result.getBestModelScore())) {
			LOGGER.debug("The best score was NaN - so the model should be bad - throwing an exception");
			throw new IllegalArgumentException("The "+getName() + " model could not be fitted using the current parameters - please revise them");
		}

		model = result.getBestModel();
		// Set the labels
		if (isClassification)
			model.setLabels(trainConv.getOneHotMapping().getLabelsND());
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

	public String toString() {
		return getName();
	}

}
