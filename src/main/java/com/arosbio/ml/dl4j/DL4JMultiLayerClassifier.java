package com.arosbio.ml.dl4j;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
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
import com.arosbio.modeling.ml.algorithms.MultiLabelClassifier;
import com.arosbio.modeling.ml.algorithms.ScoringClassifier;

public class DL4JMultiLayerClassifier extends DL4JMultiLayerBase implements ScoringClassifier, MultiLabelClassifier, Closeable {

	private static final Logger LOGGER = LoggerFactory.getLogger(DL4JMultiLayerClassifier.class);

	public static final String NAME = "DL4JMultiLayerClassifier";
	public static final String DESCRIPTION = "A Deeplearning/Artifical Neural Network (DL/ANN) for classification, implemented in Deeplearning4J";
	public static final int ID = 17;
	public static final LossFunction DEFAULT_LOSS_FUNC = LossFunction.MCXENT;

	// Settings
//	private long seed = CPSignSettings.getInstance().getRNGSeed();
	/** The width of the hidden layers in the network - all will have the same width */
//	private int networkWidth = 3;
//	private int numHiddenLayers = 3;
//	private int numEpoch = 10;
//	private Integer batchSize; // = 50;
//	private boolean saveUpdater = false;
//	private NeuralNetConfiguration.Builder config;
//	private LossFunctions.LossFunction loss = LossFunctions.LossFunction.MCXENT;
	
	// Not required to save
//	private transient int printInterval = -1;
//	private transient DataType dType = DataType.FLOAT;
	private transient int inputWidth = -1;

	// Only != null when model has been trained
	private MultiLayerNetwork model;

	public DL4JMultiLayerClassifier() {
		super();
		setLossFunc(DEFAULT_LOSS_FUNC);

//		config = new NeuralNetConfiguration.Builder()
//				.activation(Activation.RELU)
//				.weightInit(WeightInit.XAVIER)
//				.updater(new Sgd(0.9))
//				.l2(1e-4);
	}

	public DL4JMultiLayerClassifier(NeuralNetConfiguration.Builder config) {
		super(config);
		setLossFunc(DEFAULT_LOSS_FUNC);
	}
	
//	public DL4JMultiLayerClassifier setNumEpoch(int nEpoch) {
//		this.numEpoch = nEpoch;
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setBatchSize(int batchSize) {
//		this.batchSize = batchSize;
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setNumHiddenLayers(int nLayers) {
//		this.numHiddenLayers = nLayers;
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setNetworkWidth(int width) {
//		this.networkWidth = width;
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setLossFunc(LossFunction loss) {
//		this.loss = loss;
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setLoggingInterval(int interval) {
//		this.printInterval = interval;
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setUpdater(IUpdater updater) {
//		this.config.updater(updater);
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setActivation(IActivation activation) {
//		this.config.activation(activation);
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setActivation(Activation activation) {
//		this.config.activation(activation);
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setWeightInit(WeightInit init) {
//		this.config.weightInit(init);
//		return this;
//	}
//	
//	public DL4JMultiLayerClassifier setDType(DataType type) {
//		this.dType = type;
//		return this;
//	}
	
	@Override
	public Map<String, Object> getProperties() {
		// TODO Auto-generated method stub
		return new HashMap<>();
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
		INDArray labels = model.getLabels();
		Map<Integer,Double> predMapping = new HashMap<>();
		for (int i=0; i<labels.columns(); i++) {
			predMapping.put(labels.getInt(0,i), pred.getDouble(0,i));
		}
		System.err.print(labels + " pred: " + pred);
		return predMapping;
	}

	@Override
	public void train(List<DataRecord> trainingset) throws IllegalArgumentException {
		if (numHiddenLayers < 1)
			throw new IllegalStateException("Number of network layers must be at least 1");

		inputWidth = DataUtils.getMaxFeatureIndex(trainingset)+1;
		int numOutputs = DataUtils.countLabels(trainingset).size();

		// Create the list builder and add the input layer
		ListBuilder listBldr = config.seed(seed).dataType(dType).list()
				.layer(new DenseLayer.Builder().nIn(inputWidth).nOut(networkWidth).build());

		// Add hidden layers
		for (int i=0; i<numHiddenLayers; i++) {
			listBldr.layer(new DenseLayer.Builder().nIn(networkWidth).nOut(networkWidth)
					.build());
		}

		// Add output layer
		listBldr.layer( new OutputLayer.Builder(loss) 
				.activation(Activation.SOFTMAX)
				.nIn(networkWidth).nOut(numOutputs).build());

		// Create the network
		MultiLayerConfiguration netConfig = listBldr.build();
		model = new MultiLayerNetwork(netConfig);
		model.init();
		if (printInterval > 0)
			model.setListeners(new EpochScoreListener(printInterval));

		// calculate batch size 
		int batch = calcBatchSize(trainingset.size());

		DataConverter conveter = DataConverter.classification(trainingset);
		DataSetIterator iter = new INDArrayDataSetIterator(conveter, batch);

		//		RecordReaderDataSetIterator iter = new RecordRnull;

		model.fit(iter, numEpoch);
		model.setLabels(conveter.getOneHotMapping().getLabelsND());
		
//		System.err.println("nParams: " + model.numParams());
//		org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer inputLayer = (org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer) model.getLayer(0);
////		System.err.println("getNIn(): " + inputLayer.getNIn());
//		System.err.println("inputLayer: " +inputLayer);
////		System.err.println("inputLayer.batchSize: " +inputLayer.batchSize());
//		System.err.println("inputLayer.numParams: " +inputLayer.numParams());
//		System.err.println("inputLayer.getConf.variables: " +inputLayer.getConf().variables());
//		System.err.println("inputLayer.getParam(W): " +inputLayer.getParam("W"));
//		inputLayer.hasBias()
//		System.err.println("inputLayer.numParams: " +inputLayer.numParams());
//		System.err.println("inputLayer.params: " +inputLayer.params());
		
//		System.err.println("Layer toString: "+ model.getLayer(0).toString());

//		try {
//			generateVisuals(model, iter, (DataSetIterator) null);
//		} catch (Exception e) {
//			throw new IllegalArgumentException(e);
//		}
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

	/**
	 * Closes the underlying network and frees all resources
	 */
	public void close() {
		if (model != null)
			model.close();
	}


	@Override
	public DL4JMultiLayerClassifier clone() {
		DL4JMultiLayerClassifier clone = new DL4JMultiLayerClassifier(null);
		super.copyParametersToNew(clone);
		return clone;
	}

	

	

}
