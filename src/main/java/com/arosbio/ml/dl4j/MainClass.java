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

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.net.URL;
import java.util.Enumeration;
import java.util.Properties;

public class MainClass {

	private final static String PROPERTIES_FILE = "cpsign-dl4j.properties";
	private final static String BUILD_VERSION_KEY = "build.version";
	private final static String TIME_STAMP_KEY = "build.ts";

	public static void main(String[] args) { 

		// Check if CPSignApp is accessible - then forward call and args to it
		try {
			Class<?> mainCls = Class.forName("com.arosbio.cpsign.app.CPSignApp");
			// If this didn't result in a failure, CPSign code should be accessible
			Method mainMethod = mainCls.getMethod("main", String[].class);
			mainMethod.invoke(null, (Object) args);
			System.err.println("Invoked CPSignApp");
			return;
		} catch (Exception e){
			// Not there
		}

		// Otherwise fall back to print the info about the extension 

		// Get the build version and timestamp 
		String v = null, ts = null;
		Properties p = new Properties();
		try {
			Enumeration<URL> resources = MainClass.class.getClassLoader()
					.getResources(PROPERTIES_FILE);
			if (resources.hasMoreElements()) {
				try (InputStream is = resources.nextElement().openStream();){
					p.load(is);
					v = p.getProperty(BUILD_VERSION_KEY, "Undefined");
					ts = p.getProperty(TIME_STAMP_KEY, "Undefined");
				} catch(IOException ioe) {} 
			}
		} catch (Exception e){}

		StringBuilder sb = new StringBuilder();
		sb.append("%nThis JAR file is an extension to the CPSign program - providing support for Deeplearning based ML models.%n");
		if (v != null)
			sb.append("Version: ").append(v).append("%n");
		if (ts != null)
			sb.append("Build time: ").append(ts).append("%n");
		sb.append("Please visit https://github.com/arosbio/cpsign-dl4j for more information%n%n");
		
		
		System.out.printf(sb.toString());
	}

}
