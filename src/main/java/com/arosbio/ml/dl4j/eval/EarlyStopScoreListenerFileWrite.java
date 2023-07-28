/*
 * Copyright (C) Aros Bio AB.
 *
 * CPSign is an Open Source Software that is dual licensed to allow you to choose a license that best suits your requirements:
 *
 * 1) GPLv3 (GNU General Public License Version 3) with Additional Terms, including an attribution clause as well as a limitation to use the software for commercial purposes.
 *
 * 2) CPSign Proprietary License that allows you to use CPSign for commercial activities, such as in a revenue-generating operation or environment, or integrate CPSign in your proprietary software without worrying about disclosing the source code of your proprietary software, which is required if you choose to use the software under GPLv3 license. See arosbio.com/cpsign/commercial-license for details.
 */
package com.arosbio.ml.dl4j.eval;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.arosbio.io.UriUtils;

public class EarlyStopScoreListenerFileWrite extends EarlyStopScoreListener {
	private static final Logger LOGGER = LoggerFactory.getLogger(EarlyStopScoreListenerFileWrite.class);
	private static final String DEFAULT_FILE_NAME ="training_scores.csv";

	public EarlyStopScoreListenerFileWrite(String file, boolean scoresBasedOnTest) throws IOException {
		super(getWriter(file), scoresBasedOnTest);
	}

	public EarlyStopScoreListenerFileWrite(String file, boolean scoresBasedOnTest, Map<String,Object> params) throws IOException {
		super(getWriter(file), scoresBasedOnTest, params);
	}

	private static FileWriter getWriter(String path) throws IOException {
		try {
			File finalFile = null;
			if (path == null || path.isEmpty()) {
				// generate standard in current dir
				finalFile = new File(new File("").getAbsoluteFile(), DEFAULT_FILE_NAME);
			} else {
				File f = new File(UriUtils.resolvePath(path));
				if (f.exists() && f.isDirectory()) {
					finalFile = new File(f, DEFAULT_FILE_NAME);
				} else if (f.exists() && f.isFile()) {
					finalFile = f;
				} else if (! f.exists()) {
					UriUtils.createParentOfFile(f);
					finalFile = f;
				} else {
					finalFile = f;
				}

			}

			return new FileWriter(finalFile, true);
		} catch (IOException e) {
			LOGGER.debug("Failed setting up trainig-scores file ({})",path,e);
			throw new IOException("Failed setting up training score output file, reason: " + e.getMessage());
		}
	}

	/*
	private synchronized static FileWriter findFile(File dir) throws IOException {
		if (dir == null) {
			// Take the working directory
			dir = new File("").getAbsoluteFile();
		}

		if (dir.exists() && dir.isFile())
			throw new IOException("Cannot write training scores in given path ("+dir+") - it exists and is a file");

		// Make parent directories
		try {
			dir.mkdirs();
		} catch (Exception e) {
			LOGGER.debug("Failed generating directory for writing training scores to",e);
			throw new IOException("Failed creating directory to write training scores to, reason: "+e.getMessage());
		}
		File[] previousFiles = dir.listFiles(new FilenameFilter() {

			@Override
			public boolean accept(File dir, String name) {
				return name.matches(DEFAULT_FILE_NAME+"_*"+DEFAULT_FILE_ENDING);
//				return false;
			}
		});

		// Find the file to write
		File toWrite = null;

		if (previousFiles.length == 0)


		return new FileWriter(file, append);
	}
	 */



}
