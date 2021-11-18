package com.arosbio.ml.nd4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import com.arosbio.ml.nd4j.ND4JUtil.DataConverter;
import com.arosbio.modeling.data.DataRecord;
import com.arosbio.modeling.data.DataUtils;
//import com.arosbio.testutils.TestUtils;
import com.arosbio.modeling.data.Dataset.SubSet;
import com.arosbio.modeling.data.FeatureVector;
import com.arosbio.modeling.data.SparseFeature;
import com.arosbio.modeling.data.SparseFeatureImpl;
import com.arosbio.modeling.data.SparseVector;
import com.arosbio.modeling.data.transform.format.MakeDenseTransformer;

import test_utils.UnitTestBase;

public class ND4JTests extends UnitTestBase {


	@Test
	public void checkOneHotConversion() {

		Map<Double,Integer> generated = ND4JUtil.toOneHot(Arrays.asList(5d,3d,1d,0d)); 
		// Expected is to get all labels in order
		Map<Double,Integer> expected = new HashMap<>();
		expected.put(0d, 0);
		expected.put(1d, 1);
		expected.put(3d, 2);
		expected.put(5d, 3);
		
		Assert.assertEquals(expected,generated);
	}

	@Test
	public void createNDArray() {
		INDArray arr = Nd4j.zeros(2,2);
		Assert.assertEquals(2, arr.rank());
		arr.putScalar(0, 1, 2.0);
		
		Assert.assertEquals(2d, arr.getDouble(0,1),0.0001);
//		System.err.println(arr);
	}

	@Test
	public void testCPSignDatasetToND4J_Class() throws Exception {

		SubSet data = getBinaryClassificationData();

		List<DataRecord> recs = data.subList(0, 10);
		DataConverter converter = DataConverter.classification(recs);
		
		// CHECK FEATURES
		// Loop records
		for (int i = 0; i<recs.size(); i ++) {
			DataRecord r = recs.get(i);
			INDArray arr = converter.getFeaturesMatrix().getRow(i);
			assertEquals(arr, r.getFeatures());
			
		}
		
		
		// CHECK LABELS
		Map<Double,Integer> lbls = DataUtils.countLabels(recs);
		
		INDArray labelsMatrix = converter.getLabelsMatrix();
		Assert.assertEquals(lbls.size(), labelsMatrix.size(1));
		List<Double> ls = new ArrayList<>(lbls.keySet());
		Collections.sort(ls);
		
		for (int i = 0; i<recs.size(); i ++) {
//			System.err.println("r.label="+recs.get(i).getLabel() + " labelMatrix.label="+labelsMatrix.getRow(i));
			double lCpsign = recs.get(i).getLabel();
			INDArray labelRow = labelsMatrix.getRow(i);
			Assert.assertEquals(1d, labelRow.sumNumber());
			Assert.assertEquals(1d, labelRow.getDouble(ls.indexOf(lCpsign)), 0.0001);
		}
		
		// Re-do the same thing but with pre-set one-hot-mapping
		List<DataRecord> dataNew = getBinaryClassificationData().subList(0, 10);
		DataConverter secondConv = ND4JUtil.DataConverter.classification(dataNew, converter.getNumAttributes(), converter.getOneHotMapping());
		
		// The converted data should be identical to the old one!
		Assert.assertEquals(converter.getLabelsMatrix(), secondConv.getLabelsMatrix());
		Assert.assertEquals(converter.getFeaturesMatrix(), secondConv.getFeaturesMatrix());
		
	}
	
	@Test
	public void testCPSign2ND4j_denseVector_reg() throws Exception {
		SubSet data = getAndrogenReceptorRegressionData();

		List<DataRecord> recs = data.subList(0, 20);
		MakeDenseTransformer mdf = new MakeDenseTransformer();
		mdf.fit(recs);
		data = mdf.transform(data);
		
		DataConverter converter = DataConverter.regression(data);
		
		for (DataRecord r : recs) {
			// Just make sure it works
//			INDArray arr = 
			ND4JUtil.toArray(r.getFeatures(), converter.getNumAttributes());
//			System.err.println(arr);
		}
	}
	
	
	@Test
	public void testCPSignDatasetToND4J_Reg() throws Exception {

		SubSet data = getAndrogenReceptorRegressionData();

		List<DataRecord> recs = data.subList(0, 20);
		DataConverter converter = DataConverter.regression(recs);
		
		// CHECK FEATURES
		// Loop records
		for (int i = 0; i<recs.size(); i ++) {
			DataRecord r = recs.get(i);
			INDArray arr = converter.getFeaturesMatrix().getRow(i);
			assertEqualsFloat(arr, r.getFeatures());
			
		}
		
		
		// CHECK LABELS - here it should be a column-vector only
		
		INDArray labelsMatrix = converter.getLabelsMatrix();
		
		Assert.assertEquals(1, labelsMatrix.size(1));
		
		for (int i = 0; i<recs.size(); i ++) {
//			System.err.println("r.label="+recs.get(i).getLabel() + " labelMatrix.label="+labelsMatrix.getRow(i));
			double lCpsign = recs.get(i).getLabel();
			double lNd4j = labelsMatrix.getDouble(i,0);
			Assert.assertEquals(lCpsign, lNd4j, 0.0001);
		}
		
		DataConverter secondConv = DataConverter.regression(recs, converter.getNumAttributes());
		// The converted data should be identical to the old one!
		Assert.assertEquals(converter.getLabelsMatrix(), secondConv.getLabelsMatrix());
		Assert.assertEquals(converter.getFeaturesMatrix(), secondConv.getFeaturesMatrix());
	}
	
	@Test
	public void testIterator() throws IOException {
		SubSet data = getBinaryClassificationData();
		
		DataConverter conveter = DataConverter.classification(data);
		DataSetIterator iter = new INDArrayDataSetIterator(conveter, 15);
		
		while (iter.hasNext()) {
			DataSet ds = iter.next();
			INDArray feats = ds.getFeatures();
			Assert.assertEquals(15, feats.rows());
			Assert.assertEquals(15, ds.getLabels().rows());
//			System.err.println(feats.shapeInfoToString());
//			System.err.println(ds.getLabels().shapeInfoToString());
		}
	}
	
	@Test
	public void testToArray() {
		List<SparseFeature> list = Arrays.asList(new SparseFeatureImpl(0, 4.5), new SparseFeatureImpl(4, 5.6));
		FeatureVector fv = new SparseVector(list);
		INDArray arr = ND4JUtil.toArray(fv, 7);
		Assert.assertEquals(1, arr.rows());
		Assert.assertEquals(7, arr.size(1));
		
//		System.err.println(arr);
	}

	public static void assertEquals(INDArray arr, FeatureVector v) {
//		Nd4j.setDataType(DataType.dou);
		// Check all features
		for (int i = 0; i < arr.size(0); i++) {
			double ind = arr.getDouble(i);
			double feat = v.getFeature(i); 
			//			if (ind != feat) {
			//			Assert.equals
			Assert.assertEquals("Features does not match (index="+i+"): "+ ind +" =/= "+feat + "\n"+arr + "\n" + v, ind, feat, 0.00001);
			//			}
		}

	}
	
	public static void assertEqualsFloat(INDArray arr, FeatureVector v) {
//		Nd4j.setDataType(DataType.dou);
		// Check all features
		for (int i = 0; i < arr.size(0); i++) {
			float ind = arr.getFloat(i);
			float feat = (float) v.getFeature(i); 
			//			if (ind != feat) {
			//			Assert.equals
			Assert.assertEquals("Features does not match (index="+i+"): "+ ind +" =/= "+feat + "\n"+arr + "\n" + v, ind, feat, 0.0001);
			//			}
		}

	}
	
//	@Test
	public void exploreNd4jAPI() {
		INDArray mat = Nd4j.eye(3);
		System.err.println(mat);
		
		INDArray row = mat.getRow(1);
		System.err.println(row);
		System.err.println(row.shapeInfoToString());
		
		
		INDArray zeros = Nd4j.zeros(10);
		System.err.println(zeros.shapeInfoToString());
		System.err.println(zeros);
		zeros.putScalar(1, 4);
		System.err.println(zeros);
		zeros = zeros.reshape(zeros.length(),1);
		System.err.println(zeros);
	}
	
	@Test
	public void checkIrisConversion() throws IOException {
		
		SubSet data = getIrisClassificationData(); 
		DataConverter conv = ND4JUtil.DataConverter.classification(data);
		
		// Features
		INDArray features = conv.getFeaturesMatrix();
		Assert.assertEquals(4, features.columns());
		
		// Labels as one-hot
		INDArray target = conv.getLabelsMatrix();
		Assert.assertEquals(3, target.columns());
		INDArray colSums = target.sum(1);
		Assert.assertEquals(Nd4j.ones(target.rows()), colSums);
		
		Assert.assertEquals(target.rows(), features.rows());
		
	}
	
}
