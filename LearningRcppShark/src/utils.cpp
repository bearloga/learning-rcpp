//
// utils.h
//			this is part of the RcppShark package (http://github.com/aydindemircioglu/RcppShark)
//
// Copyright (C) 2015 		Aydin Demircioglu <aydin.demircioglu/at/ini.rub.de>
//
// This file is part of the RcppShark library for GNU R.
// It is made available under the terms of the GNU General Public
// License, version 2, or at your option, any later version,
// incorporated herein by reference.
//
// This program is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE.  See the GNU General Public License for more
// details.
// 
// Please do not use this software to destroy or spy on people, environment or things.
// All negative use is prohibited.
//


#include "utils.h"
#include <Rcpp.h>

using namespace shark;
using namespace Rcpp;



Data<unsigned int> NumericVectorToLabels (NumericVector X, bool verbose) {
	if (verbose) Rcout  << "Converting data.. \n";
	
	// explicit rounding.. (maybe can be done better)
	std::vector<unsigned int> outputStd (X.size());
	for (unsigned int i = 0; i < X.size(); i++) {
		outputStd [i] = round (X[i]);
	}
	
	return (createDataFromRange (outputStd));
}



UnlabeledData<RealVector> NumericMatrixToUnlabeledData (NumericMatrix X, bool verbose) {
	UnlabeledData<RealVector> outputStd;
	outputStd.inputs() = NumericMatrixToDataRealVector (X, verbose);
	return (outputStd);
}



Data<RealVector> NumericMatrixToDataRealVector (NumericMatrix X, bool verbose) {
	if (verbose) Rcout  << "Converting data.. \n";
	
	std::vector<RealVector> outputStd;
	
	// probably a bit slow, but for now its ok
	unsigned int examples = X.rows();
	for (size_t e = 0; e < examples; e++) {
		NumericMatrix::Row zzrow = X( e, _);
		RealVector tmpRV (zzrow.size());
		std::copy (zzrow.begin(), zzrow.end(), tmpRV.begin());
		outputStd.push_back(tmpRV);
	}
	
	return (createDataFromRange (outputStd));
}



NumericMatrix UnlabeledDataToNumericMatrix (UnlabeledData<RealVector> X, bool verbose) {
	return (DataRealVectorToNumericMatrix (X.inputs()), verbose);
}



NumericMatrix DataRealVectorToNumericMatrix (Data<RealVector> X, bool verbose) {
	if (verbose) Rcout  << "Converting data.. \n";
	
	NumericMatrix C(X.numberOfElements(), dataDimension(X));
	
	// probably a bit slow, but for now its ok
	unsigned int examples = X.numberOfElements();
	for (size_t e = 0; e < examples; e++) {
		RealVector p = X.element(e);
		NumericMatrix::Row zzrow = C( e, _);
		std::copy (p.begin(), p.end(), zzrow.begin());
	}
	
	return (C);
}



NumericVector LabelsToNumericVector (Data<unsigned int> X, bool verbose) {
	if (verbose) Rcout  << "Converting data.. \n";

	NumericVector C(X.numberOfElements());
	
	// probably a bit slow, but for now its ok
	for (unsigned int i = 0; i < X.numberOfElements(); i++) {
		C [i] = X.element(i);
	}
	
	return (C);
}	



Data<RealVector> NumericVectorToDataRealVector (NumericVector X, bool verbose) {
	if (verbose) Rcout  << "Converting data.. \n";

	std::vector<RealVector> outputStd;
	
	// probably a bit slow, but for now its ok
	unsigned int examples = X.size();
	for (size_t e = 0; e < examples; e++) {
		RealVector tmpRV (1);
		tmpRV[0] = X[0];
		outputStd.push_back(tmpRV);
	}
	
	return (createDataFromRange (outputStd));
}	



RealVector NumericVectorToRealVector (NumericVector X, bool verbose) {
	if (verbose) Rcout  << "Converting data.. \n";
	
	// probably a bit slow, but for now its ok
	RealVector p (X.size());
	std::copy (X.begin(), X.end(), p.begin());

	return (p);
}



NumericVector RealVectorToNumericVector (RealVector X, bool verbose) {
	if (verbose) Rcout  << "Converting data.. \n";
	
	// probably a bit slow, but for now its ok
	NumericVector p (X.size());
	std::copy (X.begin(), X.end(), p.begin());
	
	return (p);
}

