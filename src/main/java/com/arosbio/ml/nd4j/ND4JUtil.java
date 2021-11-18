package com.arosbio.ml.nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.data.DataUtils;
import com.arosbio.modeling.data.FeatureVector;
import com.arosbio.modeling.data.FeatureVector.Feature;

public class ND4JUtil {
	
	public static final DataType DEFAULT_D_TYPE = DataType.FLOAT;
	
	/**
	 * Creates a 1 x <code>numFeat</code> matrix using the default DataType {@link #DEFAULT_D_TYPE}
	 * @param v The CPSign FeatureVector instance
	 * @param numFeat The number of features the vector should have
	 * @return A 1x{@code numFeat} INDArray corresponding to the input FeatureVector {@code v}
	 */
	public static INDArray toArray(FeatureVector v, int numFeat) {
		return toArray(v, numFeat, DEFAULT_D_TYPE);
	}
	
	public static INDArray toArray(FeatureVector v, int numFeat, DataType dType) {
		INDArray arr = Nd4j.zeros(dType, 1, numFeat);
		fillArray(v, arr, numFeat-1);
		return arr;
	}
	
	/**
	 * Fill values from CPSign FeatureVector into a ND4J INDArray instance 
	 * @param v The CPSign FeatureVector instance
	 * @param row A Nd4j 2dim array of the correct shape with only zeros (to be filled)
	 * @return The {@code row} that was given as input
	 */
	public static INDArray fillArray(FeatureVector v, INDArray row) {
		return fillArray(v, row, row.columns()-1);
	}
	
	public static INDArray fillArray(FeatureVector v, INDArray row, int maxColIndex) {
		for (Feature sf : v) {
			if (maxColIndex < sf.getIndex())
				break;
			row.putScalar(sf.getIndex(), sf.getValue());
		}
		return row;
	}
	
	public static Map<Double,Integer> toOneHot(Collection<Double> labels){
		List<Double> labelsList = new ArrayList<>(labels);
		Collections.sort(labelsList);
		Map<Double,Integer> oneHotMapping = new LinkedHashMap<>();
		int ind = 0;
		for (double l : labelsList)
			oneHotMapping.put(l, ind++); 
		
		return oneHotMapping;
	}
	
	public static class OneHotMapping {
		
		private final int[] oneHotMap;
		private transient Map<Integer,Integer> labelToIndex;
		
		public OneHotMapping(Collection<? extends Number> labels) {
			Set<Integer> uniqueLabels = new HashSet<>();
			for (Number n : labels) {
				if (Math.abs(n.intValue() - n.doubleValue())>0.001) {
					throw new IllegalArgumentException("Classification labels must be integer values");
				}
				uniqueLabels.add(n.intValue());
			}
			
			List<Integer> labelList = new ArrayList<>(uniqueLabels);
			Collections.sort(labelList);
			
			oneHotMap = new int[labelList.size()];
			for (int i=0;i<labelList.size(); i++)
				oneHotMap[i] = labelList.get(i);
		}
		
		public OneHotMapping(OneHotMapping orig) {
			this.oneHotMap = orig.getLabels();
		}
		
		private void calcMappingFunction() {
			labelToIndex = new HashMap<>();
			for (int i=0; i<oneHotMap.length; i++)
				labelToIndex.put(oneHotMap[i], i);
		}
		
		public int[] getLabels() {
			return Arrays.copyOf(oneHotMap, oneHotMap.length);
		}
		
		public INDArray getLabelsND() {
			return Nd4j.createFromArray(oneHotMap);
		}
		
		public int getIndexFor(Number label) {
			if (labelToIndex == null)
				calcMappingFunction();
			
			int intLabel = label.intValue();
			if (Math.abs(intLabel-label.doubleValue())>0.0001)
				throw new IllegalArgumentException("Invalid label="+label + ", labels should be integer values");
			
			return labelToIndex.get(intLabel);
		}
		
		
	}

	public static class DataConverter implements Iterable<Pair<INDArray,INDArray>>{

		private INDArray features;
		private INDArray labels;
		
		// Things to keep
		private OneHotMapping oneHotMapping;

		private DataConverter() {}
		
		public static DataConverter classification(List<DataRecord> records) {
			return classification(records, DEFAULT_D_TYPE, DEFAULT_D_TYPE);
		}

		public static DataConverter classification(List<DataRecord> records, DataType featuresDType, DataType labelsDType) {
			DataConverter dc = new DataConverter();

			int nRows = records.size();
			int nAttr = 0;
			Set<Double> foundLabels = new HashSet<>();

			// First pass to get the labels
			for (DataRecord r : records) {
				foundLabels.add(r.getLabel());
				nAttr = Math.max(nAttr, r.getMaxFeatureIndex());
			}
			
			// Create the 1-hot mapping
			dc.oneHotMapping = new OneHotMapping(foundLabels);

			// Create the INDArrays
			dc.features = Nd4j.zeros(featuresDType, nRows, nAttr+1);
			int nLabels = foundLabels.size();
			dc.labels = Nd4j.zeros(labelsDType, nRows, nLabels);


			// Second pass fills in the INDArrays
			for (int i=0; i < records.size(); i++) {
				DataRecord r = records.get(i);
				// Set label
				dc.labels.putScalar(i,dc.oneHotMapping.getIndexFor(r.getLabel()), 1);

				// Set feature array
				INDArray row = dc.features.getRow(i);
				fillArray(r.getFeatures(), row);
			}

			return dc; 
		}
		
		public static DataConverter classification(List<DataRecord> records, int numAttributes, OneHotMapping labelMapping) {
			return classification(records, numAttributes, labelMapping, DEFAULT_D_TYPE, DEFAULT_D_TYPE);
		}
		
		public static DataConverter classification(List<DataRecord> records, 
				int numAttributes, 
				OneHotMapping labelMapping, 
				DataType featuresDType, DataType labelsDType) {
			DataConverter dc = new DataConverter();

			
			
			// Copy of the given mapping
			dc.oneHotMapping = new OneHotMapping(labelMapping); 

			// Create the INDArrays
			int nRows = records.size();
			dc.features = Nd4j.zeros(featuresDType, nRows, numAttributes);
			int nLabels = dc.oneHotMapping.getLabels().length;
			dc.labels = Nd4j.zeros(labelsDType, nRows, nLabels);


			// Only one pass for previously found data
			for (int i=0; i < records.size(); i++) {
				DataRecord r = records.get(i);
				// Set label
				dc.labels.putScalar(i,dc.oneHotMapping.getIndexFor(r.getLabel()), 1);

				// Set feature array
				INDArray row = dc.features.getRow(i);
				fillArray(r.getFeatures(), row);
			}

			return dc; 
		}

		public static DataConverter regression(List<DataRecord> records) {
			return regression(records,DEFAULT_D_TYPE, DEFAULT_D_TYPE);
		}
		
		public static DataConverter regression(List<DataRecord> records, int numAttributes) {
			return regression(records,numAttributes,DEFAULT_D_TYPE, DEFAULT_D_TYPE);
		}
		
		public static DataConverter regression(List<DataRecord> records, DataType featuresDType, DataType labelsDType) {
			return regression(records, DataUtils.getMaxFeatureIndex(records)+1, featuresDType, labelsDType);
		}
		
		public static DataConverter regression(List<DataRecord> records, int numAttributes, DataType featuresDType, DataType labelsDType) {
			DataConverter dc = new DataConverter();

			// Create the INDArrays
			int nRows = records.size();
			dc.features = Nd4j.zeros(
					featuresDType,
					nRows, 
					numAttributes
					);
			dc.labels = Nd4j.zeros(labelsDType, nRows, 1);

			// Fill with values
			for (int i=0; i < records.size(); i++) {
				DataRecord r = records.get(i);
				// Set label
				dc.labels.putScalar(i, r.getLabel());

				// Set feature array
				INDArray row = dc.features.getRow(i);
				fillArray(r.getFeatures(), row);
			}

			return dc;
		}
		
		public INDArray getFeaturesMatrix() {
			return features;
		}

		public INDArray getLabelsMatrix() {
			return labels;
		}
		
		public OneHotMapping getOneHotMapping() {
			return oneHotMapping;
		}
		
		public int getNumAttributes() {
			return features.columns();
		}
		
		/**
		 * Goes through an additional data set and converts to INDArrays. For classification
		 * it uses the same one-hot-mapping to make sure the labels are correctly converted 
		 * @param recs
		 * @return X matrix and y vector (possibly as 2D)
		 */
//		public Pair<INDArray,INDArray> convertTestSet(List<DataRecord> recs){
//			if (oneHotMapping!=null) {
//				// Classification
//			} else {
//				// Regression
//			}
//		}

		@Override
		public Iterator<Pair<INDArray, INDArray>> iterator() {
			return new Iter();
		}

		private class Iter implements Iterator<Pair<INDArray, INDArray>>{

			int row = 0;

			@Override
			public boolean hasNext() {
				return row < features.rows();
			}

			@Override
			public Pair<INDArray, INDArray> next() {
				try {
					return Pair.of(features.getRow(row), labels.getRow(row));
				} finally {
					row++;
				}
			}

		}

	}

}
