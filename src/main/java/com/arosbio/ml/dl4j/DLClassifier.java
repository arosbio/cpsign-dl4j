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

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
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

import com.arosbio.data.DataRecord;
import com.arosbio.data.DataUtils;
import com.arosbio.data.FeatureVector;
import com.arosbio.ml.algorithms.MultiLabelClassifier;
import com.arosbio.ml.algorithms.PseudoProbabilisticClassifier;
import com.arosbio.ml.nd4j.ND4JUtil;
import com.google.common.collect.ImmutableList;

public class DLClassifier extends DL4JMultiLayerBase 
	implements MultiLabelClassifier, PseudoProbabilisticClassifier {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DLClassifier.class);

	public static final String NAME = "DLClassifier";
	public static final String DESCRIPTION = "A Deeplearning/Artifical Neural Network (DL/ANN) for classification, implemented in Deeplearning4J";
	public static final int ID = 17;
	public static final LossFunction DEFAULT_LOSS_FUNC = LossFunction.MCXENT;

	private transient int[] labels;

	public DLClassifier() {
		super(DEFAULT_LOSS_FUNC);
	}

	public DLClassifier(NeuralNetConfiguration.Builder config) {
		super(DEFAULT_LOSS_FUNC, config);
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
	public List<Integer> getLabels() {
		if (this.labels != null) {
			List<Integer> lst = new ArrayList<>(labels.length);
			for (int l : labels) {
				lst.add(l);
			}
			return lst;
		}
		// Empty list if no labels
		return ImmutableList.of();
	}

	private final static String LABELS_LINE_START = "labels ";
	private final static String END_OF_CUSTOM_ADDITION = "\n\n";
	private final static Charset CHARSET = StandardCharsets.UTF_8;

	@Override
	public void saveToStream(OutputStream ostream) throws IOException, IllegalStateException {
		if (model == null)
			throw new IllegalStateException("Model not trained yet");

		// Check if labels are non-standard (and need to be saved)
		boolean needSaving = false;
		for (int i = 0; i < labels.length; i++) {
			if (labels[i] != i) {
				needSaving = true;
				break;
			}
		}
		if (needSaving) {
			LOGGER.debug("Saving labels {} from {} model to stream", Arrays.toString(labels), NAME);

			try (
					BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(ostream, CHARSET));) {
				writer.write(LABELS_LINE_START);
				writer.write(Arrays.toString(labels));
				writer.write(END_OF_CUSTOM_ADDITION);
				writer.flush();

				// continue to save in base-class, which saves the full model
				super.saveToStream(ostream);
			}
		} else {
			LOGGER.debug("Model have standard labels, not required to be saved");
			super.saveToStream(ostream);
		}

	}

	@Override
	public void loadFromStream(InputStream istream) throws IOException {
		LOGGER.debug("Loading labels from input in ml model {}", NAME);
		try (
				BufferedInputStream buff = new BufferedInputStream(istream);) {

			// Mark the stream - to check if custom labels have been added to the
			// serialization
			buff.mark(150);

			byte[] checkStart = new byte[LABELS_LINE_START.getBytes(CHARSET).length];
			buff.read(checkStart);
			String txt = new String(checkStart, CHARSET);
			if (txt.equals(LABELS_LINE_START)) {
				// have have added the custom labels to the serialized format

				// Here we should have read the inital part only, and stand at the start of the
				// [...] list

				int numBytesToTest = 50;
				buff.mark(numBytesToTest);
				// Read until END_OF_CUSTOM_ADDITION is found
				byte[] toRead = new byte[numBytesToTest];
				buff.read(toRead);
				String readTxt = new String(toRead, CHARSET);
				String[] listTxt = readTxt.split(END_OF_CUSTOM_ADDITION);
				if (listTxt.length < 2) {
					// Failed finding labels!
					LOGGER.error("Could not find labels from serialized model, read '{}'", readTxt);
					throw new IOException("fail 1");
				}

				// Convert to labels
				String labelsTxt = listTxt[0].substring(1, listTxt[0].length() - 1); // Skip the "[]" stuff
				LOGGER.debug("parsed out this as the list of labels: {}", labelsTxt);
				String[] labelsSplits = labelsTxt.split(",");
				labels = new int[labelsSplits.length];
				for (int i = 0; i < labelsSplits.length; i++) {
					labels[i] = Integer.parseInt(labelsSplits[i].trim());
				}
				LOGGER.debug("Loaded labels {} from stream", Arrays.asList(labels));

				// Here we need to reset to correct location in the stream
				buff.reset(); // here we're after the first thing (LABELS_LINE_START)
				// Calculate how many bytes to skip before DL4j model starts
				int nBytes = (listTxt[0] + END_OF_CUSTOM_ADDITION).getBytes(CHARSET).length;
				buff.skip(nBytes);
				buff.mark(-1); // Invalidate the mark

				// Continue to load from Dl4j
				super.loadFromStream(buff);

			} else {
				// Here we haven't added custom labels to the serialized format - reset and read
				// fully from
				// ModelSerializer method of dl4j
				buff.reset();
				super.loadFromStream(buff);

				// This means that we have no labels at this point and have to create the
				// default one instead
				// Labels are always ordered and cpsign give them 0,1,2,... by default
				labels = new int[model.getOutputLayer().getParam("b").columns()];
				for (int l = 0; l < labels.length; l++) {
					labels[l] = l;
				}

			}

		}

	}

	@Override
	public int predictClass(FeatureVector vector) throws IllegalStateException {
		if (model == null)
			throw new IllegalStateException("Model not trained yet");
		return labels[model.predict(ND4JUtil.toArray(vector, getInputWidth()))[0]];
	}

	@Override
	public Map<Integer, Double> predictScores(FeatureVector vector)
			throws IllegalStateException {
		if (model == null)
			throw new IllegalStateException("Model not trained yet");
		INDArray pred = model.output(ND4JUtil.toArray(vector, getInputWidth()));

		Map<Integer, Double> predMapping = new HashMap<>();
		for (int i = 0; i < labels.length; i++) {
			predMapping.put(labels[i], pred.getDouble(0, i));
		}
		return predMapping;
	}

	@Override
	public Map<Integer, Double> predictProbabilities(FeatureVector vector) throws IllegalStateException {
		return predictScores(vector);
	}
	
	@Override
	public void fit(List<DataRecord> trainingset) throws IllegalArgumentException {
		train(trainingset);
	}

	@Override
	public void train(List<DataRecord> trainingset) throws IllegalArgumentException {

		int nInput = DataUtils.getMaxFeatureIndex(trainingset) + 1;
		int numClasses = DataUtils.countLabels(trainingset).size();

		// Create the list builder and add the input layer
		ListBuilder listBldr = config.seed(getSeed()).dataType(dType).list();

		// Add hidden layers
		int lastW = addHiddenLayers(listBldr, nInput);

		// Add output layer
		listBldr.layer(new OutputLayer.Builder(loss)
				.activation(Activation.SOFTMAX)
				.nIn(lastW).nOut(numClasses)
				.dropOut(1) // no drop out for the output layer
				.build());

		trainNetwork(listBldr.build(), trainingset, true);

		// Set labels
		labels = model.getLabels().toIntVector();

	}

	@Override
	public DLClassifier clone() {
		DLClassifier clone = new DLClassifier(null);
		super.copyParametersToNew(clone);
		return clone;
	}

}
